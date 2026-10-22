"""
trainer.py
----------
Training and evaluation logic for Deep MMMF.

Based on the paper:
  "Deep Maximum Margin Matrix Factorization"
  Scientific Reports (2026) 16:14518

Training Strategy:
------------------
  Weak Generalization:
    - All users are seen during training; only some ratings are held out.
    - The full model (all parameters) is trained end-to-end.

  Strong Generalization:
    - A subset of users is COMPLETELY HELD OUT during training.
    - Phase 1: Train a base model on training users (all parameters).
    - Phase 2: Freeze item embeddings + item transform; train only user parameters
               for the held-out test users using their observed ratings.

Optimizer: Adam (paper, Table 2).
Loss: All-threshold smooth hinge loss (Eq. 3 from the paper).
Evaluation Metric: NMAE = MAE / normalization_factor (primary metric in the paper).

Key functions:
  - train_deep_mmmf()         : Main training loop (weak generalization)
  - train_new_users()         : Strong generalization user fine-tuning
  - evaluate_nmae()           : Compute NMAE metric (Table 3 in the paper)
  - predict_ratings_batched() : Batched inference to avoid OOM on large datasets
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
#  NORMALIZATION FACTOR FOR NMAE  (Section "Evaluation metrics")
# ============================================================

# NMAE normalization factors from the paper:
#   MovieLens  (R=5, ratings {1,2,3,4,5}): factor = 1.6
#   EachMovie  (ratings {0,.2,.4,...,1}):  factor = 1.94428
#   Prime Pantry (R=5):                     factor = 1.6
# NMAE = MAE / factor
NMAE_FACTORS = {
    5: 1.6,       # 5-star scale (MovieLens, Prime Pantry)
    6: 1.94428,   # EachMovie (6 levels: 0, 0.2, 0.4, 0.6, 0.8, 1.0)
}


def get_nmae_factor(num_ratings):
    """
    Return the NMAE normalization factor for a given number of rating levels.

    The factor represents the expected MAE when ratings and predictions
    are uniformly distributed. Dividing MAE by this factor gives NMAE,
    which is comparable across datasets with different rating scales.

    Parameters:
        num_ratings (int): Number of discrete rating levels R.

    Returns:
        float: Normalization factor for NMAE computation.
    """
    return NMAE_FACTORS.get(num_ratings, 1.0)


# ============================================================
#  TRAINING FUNCTION -- WEAK GENERALIZATION (End-to-End)
# ============================================================

def train_deep_mmmf(model, train_user_ids, train_item_ids, train_ratings,
                    val_user_ids, val_item_ids, val_ratings,
                    epochs, batch_size, learning_rate, lambda_reg,
                    device, verbose=True):
    """
    Train the Deep MMMF model end-to-end (weak generalization setting).

    The model is trained on (user_id, item_id, rating) triplets from the
    training set. The all-threshold smooth hinge loss (Eq. 3) is minimized
    over all observed (user, item) pairs and all R-1 threshold levels.

    Optimizer: Adam (paper, Table 2) with the specified learning rate.
    L2 regularization is applied inside compute_loss() as part of the loss.

    Parameters:
        model         : DeepMMMF model instance
        train_user_ids (np.array): Shape (N_train,) -- training user indices
        train_item_ids (np.array): Shape (N_train,) -- training item indices
        train_ratings  (np.array): Shape (N_train,) -- training ordinal ratings
        val_user_ids   (np.array): Shape (N_val,)   -- validation user indices
        val_item_ids   (np.array): Shape (N_val,)   -- validation item indices
        val_ratings    (np.array): Shape (N_val,)   -- validation ordinal ratings
        epochs        (int)  : Number of full passes over the training data.
        batch_size    (int)  : Mini-batch size (paper: 512).
        learning_rate (float): Adam learning rate (paper uses various lr values).
        lambda_reg    (float): L2 regularization coefficient (paper: 1e-6 or 1e-4).
        device        : torch.device -- CPU or CUDA GPU.
        verbose       (bool) : Whether to print epoch-level progress.

    Returns:
        model         : Trained DeepMMMF model.
        train_losses  (list): Average training loss per epoch.
        val_nmae_list (list): Validation NMAE per epoch.
    """
    # Move model to target device (GPU if available)
    model = model.to(device)

    # Adam optimizer: adaptive learning rates per parameter.
    # Default betas (0.9, 0.999) and epsilon (1e-8) as in the paper.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine Annealing LR scheduler (Loshchilov & Hutter, "SGDR", ICLR 2017).
    # Decays LR from `learning_rate` down to `eta_min` following a cosine curve.
    # This prevents the optimizer from getting stuck in local minima late in
    # training and consistently improves both MAE and ranking metrics.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )

    # ----- Convert training numpy arrays to PyTorch tensors -----
    train_users_t   = torch.tensor(train_user_ids, dtype=torch.long)
    train_items_t   = torch.tensor(train_item_ids, dtype=torch.long)
    train_ratings_t = torch.tensor(train_ratings,  dtype=torch.long)

    # ----- Validation tensors (moved to device once, reused each epoch) -----
    val_users_t   = torch.tensor(val_user_ids, dtype=torch.long).to(device)
    val_items_t   = torch.tensor(val_item_ids, dtype=torch.long).to(device)
    val_ratings_np = val_ratings  # Keep as numpy for metric computation

    # ----- DataLoader: shuffles and batches training data each epoch -----
    train_dataset = TensorDataset(train_users_t, train_items_t, train_ratings_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)

    train_losses  = []   # Track average training loss per epoch
    val_nmae_list = []   # Track validation NMAE per epoch

    if verbose:
        print(f"\n  Training Deep MMMF for {epochs} epochs...")
        print(f"  Batch size: {batch_size} | LR: {learning_rate} | Lambda: {lambda_reg}")

    for epoch in range(epochs):
        # ==========================
        #  TRAINING PHASE
        # ==========================
        model.train()   # Enable dropout (training mode)
        epoch_loss  = 0.0
        num_batches = 0

        for batch_users, batch_items, batch_ratings in train_loader:
            # Move batch to computation device
            batch_users   = batch_users.to(device)
            batch_items   = batch_items.to(device)
            batch_ratings = batch_ratings.to(device)

            # Reset gradients (PyTorch accumulates gradients by default)
            optimizer.zero_grad()

            # Compute the Deep MMMF loss (Eq. 3 in the paper)
            # This includes: all-threshold hinge loss + L2 regularization
            loss = model.compute_loss(batch_users, batch_items, batch_ratings, lambda_reg)

            # Backpropagation: compute gradients of loss w.r.t. all parameters
            loss.backward()

            # Update parameters using Adam optimizer
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

        # Average training loss for this epoch (for logging)
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # ==========================
        #  VALIDATION PHASE
        # ==========================
        # Compute NMAE on the validation set to monitor overfitting
        model.eval()
        val_predictions_np = predict_ratings_batched(model, val_users_t, val_items_t,
                                                     batch_size=batch_size, device=device)
        val_nmae = compute_nmae(val_ratings_np, val_predictions_np, model.num_ratings)
        val_nmae_list.append(val_nmae)

        # Print progress every 10 epochs (or on the first epoch)
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch [{epoch + 1:3d}/{epochs}] "
                  f"Loss: {avg_train_loss:.4f} | Val NMAE: {val_nmae:.4f} | LR: {current_lr:.6f}")

        # Step the LR scheduler (cosine annealing update after each epoch)
        scheduler.step()

    return model, train_losses, val_nmae_list


# ============================================================
#  STRONG GENERALIZATION: FINE-TUNE NEW USERS
# ============================================================

def freeze_item_parameters(model):
    """
    Freeze item-related parameters so they are NOT updated during new-user training.

    In the strong generalization setting (paper Section "Experimental setup"):
      - Item embeddings and item transform weights are trained on the base (training) users.
      - When adapting to held-out test users, we FREEZE item parameters and only
        train user-related parameters (user embedding + user transform).

    This simulates the real-world scenario where item representations are fixed
    and only the new user's preferences are inferred.

    Parameters:
        model : DeepMMMF instance -- item-related params will be frozen.
    """
    # Freeze item embedding weights (W_V matrix)
    for param in model.item_embedding.parameters():
        param.requires_grad = False

    # Freeze item transformation network (Psi_2) weights
    for param in model.item_transform.parameters():
        param.requires_grad = False

    print("  [Strong Gen] Item embedding and item transform parameters FROZEN.")


def train_new_users(model, new_user_ids, train_item_ids, train_ratings,
                    epochs, batch_size, learning_rate, lambda_reg, device, verbose=True):
    """
    Fine-tune a pre-trained Deep MMMF model to learn representations for new users.

    This implements the strong generalization procedure from the paper:
      - Item parameters (embedding + transform) are FROZEN from the base model.
      - Only user parameters (embedding + user transform + threshold embedding) are updated.
      - The optimizer only receives parameters with requires_grad=True.

    Parameters:
        model         : Pre-trained DeepMMMF (item params already frozen via freeze_item_parameters).
        new_user_ids   (np.array): Shape (N,) -- user indices of held-out users.
        train_item_ids (np.array): Shape (N,) -- item indices observed for these users.
        train_ratings  (np.array): Shape (N,) -- observed ratings for these users.
        epochs        (int)  : Fine-tuning epochs.
        batch_size    (int)  : Mini-batch size.
        learning_rate (float): Adam learning rate.
        lambda_reg    (float): L2 regularization coefficient.
        device        : torch.device.
        verbose       (bool) : Whether to print progress.

    Returns:
        model : Fine-tuned DeepMMMF model.
    """
    model = model.to(device)

    # Only optimize parameters that are NOT frozen (user embedding, user transform, thresholds)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Convert to tensors
    users_t   = torch.tensor(new_user_ids, dtype=torch.long)
    items_t   = torch.tensor(train_item_ids, dtype=torch.long)
    ratings_t = torch.tensor(train_ratings,  dtype=torch.long)

    dataset    = TensorDataset(users_t, items_t, ratings_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"\n  [Strong Gen] Fine-tuning new user parameters for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for batch_users, batch_items, batch_ratings in dataloader:
            batch_users   = batch_users.to(device)
            batch_items   = batch_items.to(device)
            batch_ratings = batch_ratings.to(device)

            optimizer.zero_grad()

            # Same loss function as training, but gradients only flow to unfrozen params
            loss = model.compute_loss(batch_users, batch_items, batch_ratings, lambda_reg)
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"    Epoch [{epoch + 1:3d}/{epochs}] Loss: {avg_loss:.4f}")

    return model


# ============================================================
#  BATCHED PREDICTION (avoids OOM on large test sets)
# ============================================================

def predict_ratings_batched(model, user_ids_tensor, item_ids_tensor,
                             batch_size=512, device=None):
    """
    Run model inference in batches to avoid GPU out-of-memory errors on large test sets.

    Parameters:
        model            : DeepMMMF model (should be in eval mode).
        user_ids_tensor  (Tensor): Shape (N,) -- user indices (already on device or CPU).
        item_ids_tensor  (Tensor): Shape (N,) -- item indices.
        batch_size       (int)   : Inference batch size.
        device           : torch.device.

    Returns:
        np.array: Shape (N,) -- predicted ratings as float numpy array.
    """
    model.eval()
    predictions = []
    N = user_ids_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            # Slice the batch and move to device if not already there
            batch_users = user_ids_tensor[start:end]
            batch_items = item_ids_tensor[start:end]

            if device is not None:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)

            # forward() calls predict_ratings() which finds the ordinal bin
            preds = model(batch_users, batch_items)
            predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def predict_scores_batched(model, user_ids_tensor, item_ids_tensor,
                            batch_size=512, device=None):
    """
    Run batched inference returning RAW CONTINUOUS SCORES (not quantised integers).

    This is the companion to predict_ratings_batched(). The raw dot-product
    score s_tilde_{ij} = U_tilde_i · V_tilde_j gives a continuous preference
    value for every (user, item) pair, which is essential for ranking metrics:

      - Integer predictions (1-5) cause ties — many items get the same score,
        making their relative order arbitrary → poor F1, MAP, MRR, FCP.
      - Continuous scores uniquely differentiate every item → accurate ranking.

    Use this output as the `ranking_scores` argument to evaluate_model().

    Parameters:
        model            : DeepMMMF model (should be in eval mode).
        user_ids_tensor  (Tensor): Shape (N,) -- user indices.
        item_ids_tensor  (Tensor): Shape (N,) -- item indices.
        batch_size       (int)   : Inference batch size.
        device           : torch.device.

    Returns:
        np.array: Shape (N,) -- continuous interaction scores.
    """
    model.eval()
    all_scores = []
    N = user_ids_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_users = user_ids_tensor[start:end]
            batch_items = item_ids_tensor[start:end]
            if device is not None:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
            scores = model.predict_scores(batch_users, batch_items)
            all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


# ============================================================
#  EVALUATION METRICS
# ============================================================

def compute_mae(true_ratings, predicted_ratings):
    """
    Compute Mean Absolute Error (MAE).

    MAE = (1/N) * sum(|y_true - y_pred|)

    This is the base metric used inside NMAE computation.

    Parameters:
        true_ratings      (np.array): Ground truth ordinal ratings.
        predicted_ratings (np.array): Model predicted ratings.

    Returns:
        float: MAE value (lower is better).
    """
    return float(np.mean(np.abs(true_ratings.astype(float) - predicted_ratings.astype(float))))


def compute_nmae(true_ratings, predicted_ratings, num_ratings):
    """
    Compute Normalized Mean Absolute Error (NMAE).

    NMAE is the primary evaluation metric in the Deep MMMF paper (Table 3).
    It normalizes MAE by a factor that represents the expected MAE under
    a uniform random prediction baseline, making it comparable across datasets.

    NMAE = MAE / normalization_factor

    Normalization factors (from the paper):
        MovieLens  (R=5): factor = 1.6
        EachMovie  (R=6): factor = 1.94428
        Prime Pantry (R=5): factor = 1.6

    Lower NMAE is better. NMAE < 1.0 means the model is better than random.

    Parameters:
        true_ratings      (np.array): Ground truth ratings.
        predicted_ratings (np.array): Predicted ratings.
        num_ratings       (int)     : Number of discrete rating levels R.

    Returns:
        float: NMAE value (lower is better).
    """
    mae    = compute_mae(true_ratings, predicted_ratings)
    factor = get_nmae_factor(num_ratings)
    return mae / factor


def compute_ranking_metrics(user_ids, true_ratings, predicted_ratings,
                             top_k=10, threshold=3.5):
    """
    Compute per-user ranking metrics: F1, F2 (F-beta), FCP, MAP, MRR.

    These match the evaluation protocol of the TripAdvisor multi-criteria
    collaborative filtering paper (Nassar et al., 2019), enabling direct
    comparison between Deep MMMF and the TripAdvisor DNN model.

    Metric definitions:
        F1  : F1-score (beta=1) at top-K -- balances precision and recall
        F2  : F2-score (beta=2) at top-K -- recall-heavy F-measure
        FCP : Fraction of Concordant Pairs (like Kendall's tau, per user)
        MAP : Mean Average Precision across users
        MRR : Mean Reciprocal Rank across users

    An item is considered 'relevant' if its true rating >= threshold.

    Parameters:
        user_ids          (np.array): Shape (N,) -- user index per sample.
        true_ratings      (np.array): Shape (N,) -- ground truth ratings.
        predicted_ratings (np.array): Shape (N,) -- predicted ratings.
        top_k             (int)     : K for top-K ranking metrics.
        threshold         (float)   : Relevance threshold (default 3.5).

    Returns:
        dict: {'F1', 'F2', 'FCP', 'MAP', 'MRR'} -- mean values across users.
    """
    from collections import defaultdict

    # Group samples by user
    user_data = defaultdict(lambda: {'true': [], 'pred': []})
    for u, t, p in zip(user_ids, true_ratings, predicted_ratings):
        user_data[u]['true'].append(float(t))
        user_data[u]['pred'].append(float(p))

    f1_list, f2_list, fcp_list, ap_list, rr_list = [], [], [], [], []

    for _, data in user_data.items():
        true = np.array(data['true'])
        pred = np.array(data['pred'])
        n    = len(true)

        # ---- Rank items by predicted rating (descending) ----
        sorted_idx     = np.argsort(-pred)
        true_sorted    = true[sorted_idx]
        relevant       = (true >= threshold).astype(int)
        rel_sorted     = relevant[sorted_idx]
        n_relevant     = int(relevant.sum())

        if n_relevant > 0:
            k = min(top_k, n)
            topk_rel = rel_sorted[:k]
            n_hit    = int(topk_rel.sum())

            # ---- Precision@K and Recall@K ----
            precision = n_hit / k          if k > 0          else 0.0
            recall    = n_hit / n_relevant if n_relevant > 0 else 0.0

            # ---- F1 (beta=1) ----
            denom_f1 = precision + recall
            f1 = 2.0 * precision * recall / denom_f1 if denom_f1 > 0 else 0.0
            f1_list.append(f1)

            # ---- F2 (beta=2, recall-heavy) ----
            beta2     = 4.0   # beta^2
            denom_f2  = beta2 * precision + recall
            f2 = (1 + beta2) * precision * recall / denom_f2 if denom_f2 > 0 else 0.0
            f2_list.append(f2)

            # ---- AP: Average Precision (for MAP, Eq. 19) ----
            ap   = 0.0
            hits = 0
            for i, rel in enumerate(rel_sorted):
                if rel:
                    hits += 1
                    ap   += hits / (i + 1)
            norm_factor = min(k, n_relevant) if min(k, n_relevant) > 0 else 1
            ap_list.append(ap / norm_factor)

            # ---- RR: Reciprocal Rank (for MRR, Eq. 20) ----
            rr = 0.0
            for i, rel in enumerate(rel_sorted):
                if rel:
                    rr = 1.0 / (i + 1)
                    break
            rr_list.append(rr)

        # ---- FCP: Fraction of Concordant Pairs (vectorised, Eqs. 14-18) ----
        # For all unique pairs (i, j), a pair is concordant if both
        # true and predicted orderings agree (sign match).
        if n > 1:
            diff_true = true[:, None] - true[None, :]   # (n, n)
            diff_pred = pred[:, None] - pred[None, :]   # (n, n)
            mask      = np.triu(np.ones((n, n), dtype=bool), k=1)
            sign_prod = diff_true[mask] * diff_pred[mask]
            concordant    = int((sign_prod > 0).sum())
            discordant    = int((sign_prod < 0).sum())
            total_pairs   = concordant + discordant
            if total_pairs > 0:
                fcp = concordant / total_pairs
                fcp_list.append(fcp)

    return {
        'F1':  float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0,
        'F2':  float(np.mean(f2_list)) if len(f2_list) > 0 else 0.0,
        'FCP': float(np.mean(fcp_list)) if len(fcp_list) > 0 else 0.5,
        'MAP': float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0,
        'MRR': float(np.mean(rr_list)) if len(rr_list) > 0 else 0.0,
    }


def evaluate_model(true_ratings, predicted_ratings, num_ratings, user_ids=None,
                   ranking_scores=None, top_k=10, relevance_threshold=3.5):
    """
    Compute evaluation metrics for Deep MMMF.

    Always computes: MAE, NMAE (Deep MMMF paper, Table 3).
    If user_ids provided: also computes F1, F2, FCP, MAP, MRR
    (TripAdvisor paper, Nassar et al. 2019, Table 6) for direct comparison.

    IMPORTANT - Two prediction arrays:
      - predicted_ratings : integer predictions (1-5)  → used for MAE/NMAE
      - ranking_scores    : continuous dot-product scores → used for ranking metrics

    Using continuous scores for ranking eliminates the tie problem: when many
    items receive the same integer rating, their ordering is random. Continuous
    scores differentiate every item and dramatically improve F1, MAP, MRR, FCP.

    Parameters:
        true_ratings       (np.array): Ground truth overall ratings.
        predicted_ratings  (np.array): Integer predicted ratings for MAE/NMAE.
        num_ratings        (int)     : Number of discrete rating levels R.
        user_ids           (np.array): Optional -- user index per sample.
        ranking_scores     (np.array): Optional -- continuous scores for ranking.
                                       Falls back to predicted_ratings if None.
        top_k              (int)     : K for top-K ranking metrics.
        relevance_threshold(float)   : Threshold above which an item is relevant.

    Returns:
        dict: Metric dictionary (always has 'MAE', 'NMAE';
              also 'F1', 'F2', 'FCP', 'MAP', 'MRR' when user_ids given).
    """
    predicted_ratings = np.clip(predicted_ratings, 1, num_ratings)

    metrics = {
        'MAE':  compute_mae(true_ratings, predicted_ratings),
        'NMAE': compute_nmae(true_ratings, predicted_ratings, num_ratings),
    }

    if user_ids is not None:
        # Use continuous scores for ranking if available, else fall back to integers
        scores_for_ranking = ranking_scores if ranking_scores is not None else predicted_ratings
        ranking = compute_ranking_metrics(
            user_ids, true_ratings, scores_for_ranking,
            top_k=top_k, threshold=relevance_threshold
        )
        metrics.update(ranking)

    return metrics


# ============================================================
#  MULTI-CRITERIA DEEP MMMF TRAINING & EVALUATION
# ============================================================

def train_multi_criteria_deep_mmmf(model, train_user_ids, train_item_ids,
                                    train_criteria_ratings, train_overall_ratings,
                                    val_user_ids, val_item_ids,
                                    val_criteria_ratings, val_overall_ratings,
                                    epochs, batch_size, learning_rate, lambda_reg,
                                    device, verbose=True):
    """
    Train the MultiCriteriaDeepMMMF model end-to-end.

    Jointly trains independent factorization paths for all 7 criteria matrices
    with coupled threshold embeddings and the final ranking aggregator.

    Parameters:
        model                  : MultiCriteriaDeepMMMF model instance
        train_user_ids (np.array)      : Shape (N_train,)
        train_item_ids (np.array)      : Shape (N_train,)
        train_criteria_ratings (np.array): Shape (N_train, 7)
        train_overall_ratings (np.array): Shape (N_train,)
        val_user_ids (np.array)        : Shape (N_val,)
        val_item_ids (np.array)        : Shape (N_val,)
        val_criteria_ratings (np.array): Shape (N_val, 7)
        val_overall_ratings (np.array) : Shape (N_val,)
        epochs (int)                   : Training epochs
        batch_size (int)               : Mini-batch size
        learning_rate (float)          : Adam learning rate
        lambda_reg (float)             : L2 regularization
        device                         : torch.device
        verbose (bool)                 : Verbose output

    Returns:
        model         : Trained MultiCriteriaDeepMMMF
        train_losses  : List of average training losses per epoch
        val_nmae_list : List of validation overall NMAEs per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )

    # Convert training data to tensors
    train_users_t   = torch.tensor(train_user_ids, dtype=torch.long)
    train_items_t   = torch.tensor(train_item_ids, dtype=torch.long)
    train_crit_t    = torch.tensor(train_criteria_ratings, dtype=torch.float32)
    train_overall_t = torch.tensor(train_overall_ratings, dtype=torch.long)

    # Validation tensors
    val_users_t   = torch.tensor(val_user_ids, dtype=torch.long).to(device)
    val_items_t   = torch.tensor(val_item_ids, dtype=torch.long).to(device)
    val_overall_np = val_overall_ratings

    train_dataset = TensorDataset(train_users_t, train_items_t, train_crit_t, train_overall_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)

    train_losses  = []
    val_nmae_list = []

    if verbose:
        print(f"\n  Training Multi-Criteria Deep MMMF for {epochs} epochs...")
        print(f"  Criteria count: {model.num_criteria} | Batch size: {batch_size} | LR: {learning_rate}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for b_users, b_items, b_crit, b_overall in train_loader:
            b_users   = b_users.to(device)
            b_items   = b_items.to(device)
            b_crit    = b_crit.to(device)
            b_overall = b_overall.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(b_users, b_items, b_crit, b_overall, lambda_reg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)

        # Validation evaluation
        model.eval()
        val_preds = predict_mc_ratings_batched(model, val_users_t, val_items_t,
                                               batch_size=batch_size, device=device)
        val_nmae = compute_nmae(val_overall_np, val_preds, model.num_ratings)
        val_nmae_list.append(val_nmae)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch [{epoch + 1:3d}/{epochs}] "
                  f"Loss: {avg_loss:.4f} | Val NMAE: {val_nmae:.4f} | LR: {current_lr:.6f}")

        scheduler.step()

    return model, train_losses, val_nmae_list


def predict_mc_ratings_batched(model, user_ids_tensor, item_ids_tensor,
                               batch_size=512, device=None):
    """Batched prediction of overall integer ratings for MultiCriteriaDeepMMMF."""
    model.eval()
    all_preds = []
    N = user_ids_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_users = user_ids_tensor[start:end]
            b_items = item_ids_tensor[start:end]
            if device is not None:
                b_users = b_users.to(device)
                b_items = b_items.to(device)
            preds = model.predict_ratings(b_users, b_items)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def predict_mc_scores_batched(model, user_ids_tensor, item_ids_tensor,
                              batch_size=512, device=None):
    """Batched prediction of continuous overall ranking scores for MultiCriteriaDeepMMMF."""
    model.eval()
    all_scores = []
    N = user_ids_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_users = user_ids_tensor[start:end]
            b_items = item_ids_tensor[start:end]
            if device is not None:
                b_users = b_users.to(device)
                b_items = b_items.to(device)
            scores = model.predict_scores(b_users, b_items)
            all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


def predict_mc_criteria_ratings_batched(model, user_ids_tensor, item_ids_tensor,
                                        batch_size=512, device=None):
    """Batched prediction of the 7 criteria ratings for MultiCriteriaDeepMMMF."""
    model.eval()
    all_preds = []
    N = user_ids_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_users = user_ids_tensor[start:end]
            b_items = item_ids_tensor[start:end]
            if device is not None:
                b_users = b_users.to(device)
                b_items = b_items.to(device)
            preds = model.predict_criteria_ratings(b_users, b_items)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def evaluate_multi_criteria_model(test_criteria, pred_criteria, test_overall, pred_overall,
                                  num_ratings, user_ids=None, ranking_scores=None,
                                  criteria_names=None):
    """
    Comprehensive evaluation of MultiCriteriaDeepMMMF.

    Computes:
      - Overall rating MAE and NMAE
      - Ranking metrics: F1, F2, FCP, MAP, MRR
      - Per-criterion MAE for each of the 7 criteria
      - Average criteria MAE across all 7 criteria
    """
    if criteria_names is None:
        criteria_names = ['value', 'rooms', 'location', 'cleanliness', 'checkin', 'service', 'business_service']

    # 1. Overall metrics
    overall_metrics = evaluate_model(
        test_overall, pred_overall, num_ratings,
        user_ids=user_ids, ranking_scores=ranking_scores
    )

    # 2. Individual criteria MAEs
    crit_maes = {}
    crit_mae_list = []
    for c_idx, c_name in enumerate(criteria_names):
        c_true = test_criteria[:, c_idx]
        c_pred = pred_criteria[:, c_idx]
        c_mae = compute_mae(c_true, c_pred)
        crit_maes[f"MAE_{c_name}"] = c_mae
        crit_mae_list.append(c_mae)

    overall_metrics['Criteria_MAE_Mean'] = float(np.mean(crit_mae_list))
    overall_metrics.update(crit_maes)
    return overall_metrics

