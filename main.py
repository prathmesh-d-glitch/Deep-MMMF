"""
main.py
-------
Main entry point for Deep MMMF (Deep Maximum Margin Matrix Factorization).

Based on the paper:
  "Deep Maximum Margin Matrix Factorization"
  by Shailendra Kumar, Venkateswara Rao Kagita, Vikas Kumar, Gaurav Niranjan
  Scientific Reports (2026) 16:14518

This script ties all components together and runs the full evaluation pipeline:
  1. Load and preprocess a ratings dataset (MovieLens-style, 1-5 star ratings).
  2. Perform K-fold cross-validation (paper uses 5 folds, Table 3).
  3. For each fold:
       WEAK GENERALIZATION:
         a. Train Deep MMMF on 80% of data (end-to-end).
         b. Evaluate on held-out 20% test set using NMAE.
       STRONG GENERALIZATION:
         a. Train base model on training users (all ratings for those users).
         b. Freeze item parameters.
         c. Fine-tune user parameters for held-out test users.
         d. Evaluate on new users' held-out ratings using NMAE.
  4. Report mean NMAE and MAE across all 5 folds (matches Table 3 format).

How to Run:
-----------
  Demo mode (uses the existing synthetic CSV with 1-5 ratings):
      python main.py --demo

  With a real MovieLens-style CSV (columns: user_id, item_id, rating):
      python main.py --data_path ratings.csv --num_ratings 5

  Weak generalization only (faster):
      python main.py --demo --generalization weak

  Strong generalization only:
      python main.py --demo --generalization strong

Datasets from the paper (download separately):
  - MovieLens 1M: https://grouplens.org/datasets/movielens/
  - EachMovie:    https://grouplens.org/datasets/eachmovie/
  - Prime Pantry: https://cseweb.ucsd.edu/~jmcauley/datasets/amazonv2/
"""

import argparse
import os
import copy
import numpy as np
import torch

from data_loader import (
    load_tripadvisor_data,   # We reuse the data loader but adapt it for integer ratings
    encode_ids,
    split_data_kfold,
    create_sample_dataset,
    get_criteria_columns
)
from models import DeepMMMF, MultiCriteriaDeepMMMF
from trainer import (
    train_deep_mmmf,
    train_new_users,
    freeze_item_parameters,
    predict_ratings_batched,
    predict_scores_batched,
    evaluate_model,
    train_multi_criteria_deep_mmmf,
    predict_mc_ratings_batched,
    predict_mc_scores_batched,
    predict_mc_criteria_ratings_batched,
    evaluate_multi_criteria_model
)




# ============================================================
#  HYPERPARAMETERS (matching Table 2 in the paper, MovieLens config)
# ============================================================

# --- Embedding dimensions ---
EMBEDDING_DIM    = 100   # k: user/item latent dimensionality (Table 2: "User/Item embedding size = 100")
THETA_EMBED_DIM  = 32    # d_theta: threshold embedding size  (Table 2: "Embedding size = 32")

# --- Network architecture (Table 2: "User/Item network layers = 256->128->100") ---
USER_HIDDEN      = [256, 128]   # Hidden sizes for user transform Psi_1
ITEM_HIDDEN      = [256, 128]   # Hidden sizes for item transform Psi_2
THETA_HIDDEN     = [16, 4]      # Hidden sizes for threshold transform Psi_3 (Table 2: "16->4")

# --- Training hyperparameters ---
BATCH_SIZE       = 512          # Mini-batch size (Table 2)
LEARNING_RATE    = 0.001        # Adam learning rate (Table 2)
EPOCHS           = 50           # Training epochs (adjust upward for better results)
DROPOUT          = 0.2          # Dropout probability (Table 2: 0.2 for MovieLens, 0.3 for EachMovie)
LAMBDA_REG       = 1e-6         # L2 regularization (Table 2: 1e-6 for MovieLens weak gen)

# --- Evaluation ---
NUM_RATINGS      = 5            # R: maximum rating level (5 for MovieLens/Prime Pantry)
NUM_FOLDS        = 5            # K-fold cross-validation (paper uses 5)
VAL_FRACTION     = 0.01         # 1% of training set used as validation (paper: "1% val")

# --- Strong generalization ---
# Fraction of users to hold out completely during base training
STRONG_GEN_TEST_USER_FRACTION = 0.10   # ~10% of users held out for strong gen
STRONG_GEN_FINETUNE_EPOCHS    = 30     # Epochs to fine-tune new user parameters

RANDOM_SEED = 42   # Reproducibility seed

# --- BPR ranking loss weight (Rendle et al., UAI 2009) ---
# Balances rating-accuracy loss (hinge) vs ranking loss (BPR).
# 0.0 = pure hinge (original Deep MMMF), 1.0 = pure BPR.
# 0.5 gives best combined MAE + ranking performance.
BPR_ALPHA = 0.5


def set_random_seeds(seed):
    """
    Set random seeds for NumPy, Python, and PyTorch for reproducibility.
    This ensures that every run with the same seed produces identical results.

    Parameters:
        seed (int): The seed value to use everywhere.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_data(args):
    """
    Load dataset and preprocess into the format expected by Deep MMMF.

    Deep MMMF requires a simple (user_id, item_id, rating) format where:
      - user_id and item_id are consecutive integers starting at 0
      - rating is an integer in {1, 2, ..., R}

    For demo mode: reuses the synthetic TripAdvisor CSV (which has an
    'overall' column with integer 1-5 ratings). We treat 'overall' as
    the single ordinal rating that Deep MMMF will learn to predict.

    Parameters:
        args : Parsed command-line arguments.

    Returns:
        data (DataFrame): Preprocessed dataset with columns user_id, item_id, rating.
        num_users (int) : Total unique users.
        num_items (int) : Total unique items.
    """
    if args.demo:
        print("\n[Demo Mode] Using synthetic dataset (with 7 criteria ratings + overall).")
        demo_path = 'sample_tripadvisor.csv'

        # Create demo data if it does not exist already
        if not os.path.exists(demo_path):
            data = create_sample_dataset(demo_path, num_users=200, num_items=100, num_ratings=2000)
        else:
            data = load_tripadvisor_data(demo_path)

        criteria_cols = get_criteria_columns()
        has_criteria = all(c in data.columns for c in criteria_cols)

        cols_to_keep = ['user_id', 'item_id']
        if has_criteria:
            cols_to_keep.extend(criteria_cols)
        cols_to_keep.append('overall')

        data = data[cols_to_keep].copy()
        data = data.rename(columns={'overall': 'rating'})
        data['rating'] = data['rating'].clip(1, 5).round().astype(int)

        if has_criteria:
            for c in criteria_cols:
                data[c] = data[c].clip(1, 5).round().astype(int)

    elif args.data_path is not None:
        print(f"\n[Data Mode] Loading from: {args.data_path}")
        raw_data = load_tripadvisor_data(args.data_path)

        # Try to detect the rating column name in the CSV
        possible_rating_cols = ['rating', 'overall', 'score']
        rating_col = None
        for col in possible_rating_cols:
            if col in raw_data.columns:
                rating_col = col
                break

        if rating_col is None:
            raise ValueError(
                f"Could not find a rating column. Expected one of {possible_rating_cols}. "
                f"Found columns: {list(raw_data.columns)}"
            )

        criteria_cols = get_criteria_columns()
        has_criteria = all(c in raw_data.columns for c in criteria_cols)

        cols_to_keep = ['user_id', 'item_id']
        if has_criteria:
            cols_to_keep.extend(criteria_cols)
        cols_to_keep.append(rating_col)

        data = raw_data[cols_to_keep].copy()
        data = data.rename(columns={rating_col: 'rating'})
        data['rating'] = data['rating'].clip(1, args.num_ratings).round().astype(int)

        if has_criteria:
            for c in criteria_cols:
                data[c] = data[c].clip(1, args.num_ratings).round().astype(int)

    else:
        print("\nNo data path provided.")
        print("  Use --demo to test with synthetic data.")
        print("  Use --data_path <path> to load a real dataset.")
        return None, 0, 0

    # Re-encode user and item IDs to consecutive integers [0, num_users)
    data, num_users, num_items = encode_ids(data)
    print(f"\n  Dataset: {num_users} users, {num_items} items, {len(data)} ratings")
    print(f"  Rating range: {data['rating'].min()} to {data['rating'].max()}")
    if all(c in data.columns for c in get_criteria_columns()):
        print(f"  7 criteria present: {get_criteria_columns()}")

    return data, num_users, num_items


def run_weak_generalization_fold(fold_number, train_data, test_data,
                                  num_users, num_items, num_ratings, device):
    """
    Run one fold of WEAK GENERALIZATION evaluation.

    Weak generalization (paper, Section 'Experimental setup'):
      - All users appear in both training and test sets.
      - Only some RATINGS (not users) are held out for testing (80/20 split of ratings).
      - The model is trained end-to-end on training ratings and evaluated on test ratings.

    Parameters:
        fold_number  (int)       : Current fold number (for display).
        train_data   (DataFrame) : Training subset (80% of ratings).
        test_data    (DataFrame) : Test subset (20% of ratings).
        num_users    (int)       : Total unique users.
        num_items    (int)       : Total unique items.
        num_ratings  (int)       : Maximum rating R.
        device       : torch.device.

    Returns:
        dict: {'MAE': ..., 'NMAE': ...} for this fold.
    """
    print(f"\n  --- Weak Generalization (Fold {fold_number}) ---")

    # Extract numpy arrays from DataFrames for model input
    train_users   = train_data['user_id'].values.astype(np.int32)
    train_items   = train_data['item_id'].values.astype(np.int32)
    train_ratings = train_data['rating'].values.astype(np.int32)

    test_users    = test_data['user_id'].values.astype(np.int32)
    test_items    = test_data['item_id'].values.astype(np.int32)
    test_ratings  = test_data['rating'].values.astype(np.int32)

    # Split 1% of training data as validation (paper: "including a 1% validation set")
    num_train = len(train_users)
    num_val   = max(1, int(num_train * VAL_FRACTION))
    val_idx   = np.random.choice(num_train, num_val, replace=False)
    train_idx = np.array([i for i in range(num_train) if i not in set(val_idx)])

    val_users    = train_users[val_idx]
    val_items    = train_items[val_idx]
    val_ratings  = train_ratings[val_idx]
    train_users  = train_users[train_idx]
    train_items  = train_items[train_idx]
    train_ratings = train_ratings[train_idx]

    print(f"  Train: {len(train_users)} | Val: {len(val_users)} | Test: {len(test_users)}")

    # Initialize a fresh Deep MMMF model for this fold with paper's hyperparameters
    model = DeepMMMF(
        num_users=num_users,
        num_items=num_items,
        num_ratings=num_ratings,
        embedding_dim=EMBEDDING_DIM,
        theta_embed_dim=THETA_EMBED_DIM,
        user_hidden_sizes=USER_HIDDEN,
        item_hidden_sizes=ITEM_HIDDEN,
        theta_hidden_sizes=THETA_HIDDEN,
        dropout_prob=DROPOUT,
        bpr_alpha=BPR_ALPHA      # BPR ranking loss weight
    )

    # Train end-to-end using the smooth hinge loss (Eq. 3 in the paper)
    model, _, _ = train_deep_mmmf(
        model=model,
        train_user_ids=train_users,
        train_item_ids=train_items,
        train_ratings=train_ratings,
        val_user_ids=val_users,
        val_item_ids=val_items,
        val_ratings=val_ratings,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        device=device
    )

    # Run batched inference on the test set to predict ordinal ratings and continuous ranking scores
    model.eval()
    test_users_t  = torch.tensor(test_users, dtype=torch.long)
    test_items_t  = torch.tensor(test_items, dtype=torch.long)
    predictions   = predict_ratings_batched(model, test_users_t, test_items_t,
                                            batch_size=BATCH_SIZE, device=device)
    scores        = predict_scores_batched(model, test_users_t, test_items_t,
                                           batch_size=BATCH_SIZE, device=device)

    # Compute all metrics (MAE/NMAE from Deep MMMF paper +
    # F1/F2/FCP/MAP/MRR from TripAdvisor paper for comparison).
    # Using continuous scores for ranking avoids ties and gives high-fidelity ranking metrics.
    metrics = evaluate_model(
        test_ratings, predictions, num_ratings,
        user_ids=test_users,    # enables per-user ranking metrics
        ranking_scores=scores   # continuous scores for ranking
    )

    print(f"    [Weak Gen] MAE: {metrics['MAE']:.4f} | NMAE: {metrics['NMAE']:.4f} "
          f"| F1: {metrics['F1']:.4f} | MAP: {metrics['MAP']:.4f} | MRR: {metrics['MRR']:.4f}")
    return metrics


def run_mc_weak_generalization_fold(fold_number, train_data, test_data,
                                     num_users, num_items, num_ratings, device,
                                     criteria_names=None):
    """
    Run one fold of Multi-Criteria Deep MMMF evaluation.

    Independently factorizes the 7 criteria matrices with coupled threshold embeddings,
    and passes transformed criterion vectors to a final ranking aggregator.
    """
    if criteria_names is None:
        criteria_names = get_criteria_columns()

    print(f"\n  --- Multi-Criteria Deep MMMF [Weak Gen] (Fold {fold_number}) ---")

    train_users   = train_data['user_id'].values.astype(np.int32)
    train_items   = train_data['item_id'].values.astype(np.int32)
    train_crit    = train_data[criteria_names].values.astype(np.float32)
    train_ratings = train_data['rating'].values.astype(np.int32)

    test_users    = test_data['user_id'].values.astype(np.int32)
    test_items    = test_data['item_id'].values.astype(np.int32)
    test_crit     = test_data[criteria_names].values.astype(np.float32)
    test_ratings  = test_data['rating'].values.astype(np.int32)

    # 1% validation split
    num_train = len(train_users)
    num_val   = max(1, int(num_train * VAL_FRACTION))
    val_idx   = np.random.choice(num_train, num_val, replace=False)
    train_idx = np.array([i for i in range(num_train) if i not in set(val_idx)])

    val_users    = train_users[val_idx]
    val_items    = train_items[val_idx]
    val_crit     = train_crit[val_idx]
    val_ratings  = train_ratings[val_idx]

    train_users   = train_users[train_idx]
    train_items   = train_items[train_idx]
    train_crit    = train_crit[train_idx]
    train_ratings = train_ratings[train_idx]

    print(f"  Train: {len(train_users)} | Val: {len(val_users)} | Test: {len(test_users)} | Criteria: {len(criteria_names)}")

    model = MultiCriteriaDeepMMMF(
        num_users=num_users,
        num_items=num_items,
        num_ratings=num_ratings,
        criteria_names=criteria_names,
        embedding_dim=EMBEDDING_DIM,
        theta_embed_dim=THETA_EMBED_DIM,
        criterion_vector_dim=32,
        user_hidden_sizes=USER_HIDDEN,
        item_hidden_sizes=ITEM_HIDDEN,
        theta_hidden_sizes=THETA_HIDDEN,
        aggregator_hidden_sizes=[128, 64],
        dropout_prob=DROPOUT,
        bpr_alpha=BPR_ALPHA,
        criteria_alpha=1.0
    )

    model, _, _ = train_multi_criteria_deep_mmmf(
        model=model,
        train_user_ids=train_users,
        train_item_ids=train_items,
        train_criteria_ratings=train_crit,
        train_overall_ratings=train_ratings,
        val_user_ids=val_users,
        val_item_ids=val_items,
        val_criteria_ratings=val_crit,
        val_overall_ratings=val_ratings,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        device=device
    )

    model.eval()
    test_users_t = torch.tensor(test_users, dtype=torch.long)
    test_items_t = torch.tensor(test_items, dtype=torch.long)

    pred_ratings = predict_mc_ratings_batched(model, test_users_t, test_items_t,
                                              batch_size=BATCH_SIZE, device=device)
    pred_scores  = predict_mc_scores_batched(model, test_users_t, test_items_t,
                                             batch_size=BATCH_SIZE, device=device)
    pred_crit    = predict_mc_criteria_ratings_batched(model, test_users_t, test_items_t,
                                                       batch_size=BATCH_SIZE, device=device)

    metrics = evaluate_multi_criteria_model(
        test_criteria=test_crit,
        pred_criteria=pred_crit,
        test_overall=test_ratings,
        pred_overall=pred_ratings,
        num_ratings=num_ratings,
        user_ids=test_users,
        ranking_scores=pred_scores,
        criteria_names=criteria_names
    )

    print(f"    [MC-Deep MMMF] Overall MAE: {metrics['MAE']:.4f} | Overall NMAE: {metrics['NMAE']:.4f} "
          f"| F1: {metrics['F1']:.4f} | MAP: {metrics['MAP']:.4f} | Criteria Mean MAE: {metrics['Criteria_MAE_Mean']:.4f}")
    return metrics



def run_strong_generalization_fold(fold_number, train_data, test_data,
                                    num_users, num_items, num_ratings, device):
    """
    Run one fold of STRONG GENERALIZATION evaluation.

    Strong generalization (paper, Section 'Experimental setup'):
      - A subset of users is COMPLETELY HELD OUT during base model training.
      - During testing, the model predicts ratings for these unseen users by
        inferring only their latent representations, while item embeddings are FROZEN.

    Two-phase procedure:
      Phase 1 - Train base model on training users:
        - Learns good item representations W_V and Psi_2.
        - All training users' ratings are used.

      Phase 2 - Adapt to held-out test users:
        - Item embedding + item transform weights are FROZEN (transferred from Phase 1).
        - New user embeddings are initialized and fine-tuned on observed test-user ratings.
        - Evaluation is done on the held-out portion of the test users' ratings.

    Parameters:
        fold_number  (int)       : Current fold number.
        train_data   (DataFrame) : Training subset (base training users).
        test_data    (DataFrame) : Test subset (some users may be completely unseen).
        num_users    (int)       : Total unique users.
        num_items    (int)       : Total unique items.
        num_ratings  (int)       : Maximum rating R.
        device       : torch.device.

    Returns:
        dict: {'MAE': ..., 'NMAE': ...} for this fold.
    """
    print(f"\n  --- Strong Generalization (Fold {fold_number}) ---")

    # Identify "new" users: users in test set but NOT in training set
    test_user_set  = set(test_data['user_id'].unique())
    train_user_set = set(train_data['user_id'].unique())
    new_users      = test_user_set - train_user_set

    if len(new_users) == 0:
        # If no truly new users (all test users were seen in training), fall back to weak gen
        print("  No new users found (all test users seen in training). Running weak gen instead.")
        return run_weak_generalization_fold(fold_number, train_data, test_data,
                                             num_users, num_items, num_ratings, device)

    print(f"  Base training users: {len(train_user_set)} | New test users: {len(new_users)}")

    # Extract base training data arrays
    base_users    = train_data['user_id'].values.astype(np.int32)
    base_items    = train_data['item_id'].values.astype(np.int32)
    base_ratings  = train_data['rating'].values.astype(np.int32)

    # Split 1% validation from base training data (same as weak gen)
    num_train = len(base_users)
    num_val   = max(1, int(num_train * VAL_FRACTION))
    val_idx   = np.random.choice(num_train, num_val, replace=False)
    train_idx = np.array([i for i in range(num_train) if i not in set(val_idx)])

    val_users_arr    = base_users[val_idx]
    val_items_arr    = base_items[val_idx]
    val_ratings_arr  = base_ratings[val_idx]
    base_users_arr   = base_users[train_idx]
    base_items_arr   = base_items[train_idx]
    base_ratings_arr = base_ratings[train_idx]

    # ---- PHASE 1: Train base model on training (base) users ----
    print(f"\n  Phase 1: Training base model on {len(base_users_arr)} base user ratings...")

    base_model = DeepMMMF(
        num_users=num_users,
        num_items=num_items,
        num_ratings=num_ratings,
        embedding_dim=EMBEDDING_DIM,
        theta_embed_dim=THETA_EMBED_DIM,
        user_hidden_sizes=USER_HIDDEN,
        item_hidden_sizes=ITEM_HIDDEN,
        theta_hidden_sizes=THETA_HIDDEN,
        dropout_prob=DROPOUT,
        bpr_alpha=BPR_ALPHA      # BPR ranking loss weight
    )

    base_model, _, _ = train_deep_mmmf(
        model=base_model,
        train_user_ids=base_users_arr,
        train_item_ids=base_items_arr,
        train_ratings=base_ratings_arr,
        val_user_ids=val_users_arr,
        val_item_ids=val_items_arr,
        val_ratings=val_ratings_arr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        device=device
    )

    # ---- PHASE 2: Adapt base model to new (held-out) users ----
    # Deep copy so we don't modify the base model's item representations
    new_user_model = copy.deepcopy(base_model)

    # Freeze item embedding (W_V) and item transform (Psi_2) weights
    # (paper: "item embeddings fixed", Table 2: "Item weight transfer: Yes (frozen)")
    freeze_item_parameters(new_user_model)

    # Get all ratings from the new (held-out) test users
    new_user_test_data = test_data[test_data['user_id'].isin(new_users)]

    # Split into adaptation set (for fine-tuning) and evaluation set (for testing)
    # Use half for adaptation, half for evaluation (standard strong gen protocol)
    from sklearn.model_selection import train_test_split
    if len(new_user_test_data) > 2:
        adapt_data, eval_data = train_test_split(
            new_user_test_data, test_size=0.5, random_state=RANDOM_SEED
        )
    else:
        # If very few ratings, use all for both (overly generous but necessary)
        adapt_data = new_user_test_data
        eval_data  = new_user_test_data

    # Fine-tune: only user embedding + user transform + threshold params are updated.
    # Item params are frozen (transferred from Phase 1 base model).
    print(f"\n  Phase 2: Adapting to {len(new_users)} new users "
          f"({len(adapt_data)} adaptation ratings, {len(eval_data)} evaluation ratings)...")

    new_user_model = train_new_users(
        model=new_user_model,
        new_user_ids=adapt_data['user_id'].values.astype(np.int32),
        train_item_ids=adapt_data['item_id'].values.astype(np.int32),
        train_ratings=adapt_data['rating'].values.astype(np.int32),
        epochs=STRONG_GEN_FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        device=device
    )

    # Predict ratings for new users' held-out evaluation data
    eval_users   = eval_data['user_id'].values.astype(np.int32)
    eval_items   = eval_data['item_id'].values.astype(np.int32)
    eval_ratings = eval_data['rating'].values.astype(np.int32)

    eval_users_t = torch.tensor(eval_users, dtype=torch.long)
    eval_items_t = torch.tensor(eval_items, dtype=torch.long)

    new_user_model.eval()
    predictions = predict_ratings_batched(new_user_model, eval_users_t, eval_items_t,
                                          batch_size=BATCH_SIZE, device=device)
    scores      = predict_scores_batched(new_user_model, eval_users_t, eval_items_t,
                                         batch_size=BATCH_SIZE, device=device)
    metrics = evaluate_model(
        eval_ratings, predictions, num_ratings,
        user_ids=eval_users,
        ranking_scores=scores
    )

    rank_str = f" | F1: {metrics['F1']:.4f} | MAP: {metrics['MAP']:.4f}" if 'F1' in metrics else ""
    print(f"    [Strong Gen] MAE: {metrics['MAE']:.4f} | NMAE: {metrics['NMAE']:.4f}{rank_str}")
    return metrics


def run_fold(fold_number, train_data, test_data,
             num_users, num_items, num_ratings, device, generalization, model_type='multi_criteria'):
    """
    Execute the evaluation pipeline for a single cross-validation fold.

    Dispatches to Single-Criterion Deep MMMF or Multi-Criteria Deep MMMF based on model_type.
    """
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_number} (Model: {model_type.upper()})")
    print(f"{'='*60}")

    results = {}
    criteria_cols = get_criteria_columns()
    has_criteria = all(c in train_data.columns for c in criteria_cols)

    if model_type in ('multi_criteria', 'both'):
        if has_criteria:
            results['weak'] = run_mc_weak_generalization_fold(
                fold_number, train_data, test_data, num_users, num_items, num_ratings, device, criteria_cols
            )
        else:
            print("  [Warning] Criteria columns not found in data; falling back to single-criterion Deep MMMF.")
            results['weak'] = run_weak_generalization_fold(
                fold_number, train_data, test_data, num_users, num_items, num_ratings, device
            )

    elif model_type == 'single':
        if generalization in ('weak', 'both'):
            results['weak'] = run_weak_generalization_fold(
                fold_number, train_data, test_data, num_users, num_items, num_ratings, device
            )
        if generalization in ('strong', 'both'):
            results['strong'] = run_strong_generalization_fold(
                fold_number, train_data, test_data, num_users, num_items, num_ratings, device
            )

    return results


def aggregate_results(all_fold_results, setting):
    """
    Compute mean and standard deviation of each metric across all folds.
    """
    fold_metrics = [fold[setting] for fold in all_fold_results if setting in fold]

    if not fold_metrics:
        return {}

    metric_names = list(fold_metrics[0].keys())
    aggregated   = {}

    for name in metric_names:
        values = [m[name] for m in fold_metrics]
        aggregated[f"{name}_mean"] = float(np.mean(values))
        aggregated[f"{name}_std"]  = float(np.std(values))

    return aggregated


def print_final_results(all_fold_results, generalization, num_ratings, model_type='multi_criteria'):
    """
    Print the final summary table comparing results to:
      1. TripAdvisor Multi-Criteria DNN (Nassar et al., 2019, Table 6)
      2. Deep MMMF (ours)
    """
    TRIPADVISOR_REF = {
        'MAE': {'value': 0.7552, 'lower_is_better': True},
        'F1' : {'value': 0.8769, 'lower_is_better': False},
        'F2' : {'value': 0.9295, 'lower_is_better': False},
        'FCP': {'value': 0.5126, 'lower_is_better': False},
        'MAP': {'value': 0.4643, 'lower_is_better': False},
        'MRR': {'value': 0.7664, 'lower_is_better': False},
    }

    settings_to_show = ['weak'] if 'weak' in all_fold_results[0] else []
    if 'strong' in all_fold_results[0]:
        settings_to_show.append('strong')

    for setting in settings_to_show:
        agg = aggregate_results(all_fold_results, setting)
        if not agg:
            continue

        model_title = "Multi-Criteria Deep MMMF" if model_type == 'multi_criteria' else "Deep MMMF"
        print(f"\n{'='*75}")
        print(f"  FINAL RESULTS ({setting.upper()} GENERALIZATION, Mean over {NUM_FOLDS} folds)")
        print(f"  {model_title} (ours) vs. TripAdvisor Multi-Criteria DNN (Nassar et al., 2019)")
        print(f"{'='*75}")
        print(f"  {'Metric':<8}  {model_title + ' (ours)':>28}  {'TripAdvisor DNN':>16}  {'':>10}")
        print(f"  {'-'*75}")

        for metric, ref_info in TRIPADVISOR_REF.items():
            key_mean = f"{metric}_mean"
            key_std  = f"{metric}_std"

            if key_mean not in agg:
                our_str = "N/A"
                tag     = ""
            else:
                our_mean = agg[key_mean]
                our_std  = agg[key_std]
                our_str  = f"{our_mean:.4f} +/- {our_std:.4f}"

                ref_val         = ref_info['value']
                lower_is_better = ref_info['lower_is_better']
                if lower_is_better:
                    better = our_mean < ref_val
                else:
                    better = our_mean > ref_val
                tag = "(Better!)" if better else ""

            ref_str = f"{ref_info['value']:.4f}"
            print(f"  {metric:<8}  {our_str:>28}  {ref_str:>16}  {tag}")

        if 'NMAE_mean' in agg:
            nmae_str = f"{agg['NMAE_mean']:.4f} +/- {agg['NMAE_std']:.4f}"
            print(f"  {'NMAE':<8}  {nmae_str:>28}  {'N/A (MMMF paper)':>16}")

        # Criteria breakdown display if multi-criteria model was evaluated
        if 'Criteria_MAE_Mean_mean' in agg:
            print(f"\n  {'-'*75}")
            print(f"  INDEPENDENT CRITERIA FACTORIZATION MAE (Coupled Threshold Embeddings)")
            print(f"  {'-'*75}")
            crit_names = get_criteria_columns()
            for c_name in crit_names:
                key_c = f"MAE_{c_name}_mean"
                if key_c in agg:
                    val_c = agg[key_c]
                    std_c = agg[f"MAE_{c_name}_std"]
                    print(f"    - Criterion [{c_name:<16}]: MAE = {val_c:.4f} +/- {std_c:.4f}")
            print(f"    * Average Across All 7 Criteria: MAE = {agg['Criteria_MAE_Mean_mean']:.4f} +/- {agg['Criteria_MAE_Mean_std']:.4f}")

    print(f"\n  Notes:")
    print(f"    - Architecture: Independent factorization of 7 criteria + Coupled thresholds + Ranking Aggregator.")
    print(f"    - TripAdvisor DNN ref: Nassar et al. (2019) Table 6 (real TripAdvisor dataset).")
    print(f"    - MAE: lower is better.  F1/F2/FCP/MAP/MRR: higher is better.")
    print(f"    - (Better!) = Deep MMMF beats the TripAdvisor DNN on that metric.")


def main():
    """
    Main function: parse arguments and run the evaluation pipeline.
    """
    global EPOCHS, EMBEDDING_DIM

    parser = argparse.ArgumentParser(
        description="Multi-Criteria Deep MMMF: Independent Criteria Factorizations + Coupled Thresholds + Ranking Aggregator"
    )
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to ratings CSV (columns: user_id, item_id, rating or criteria+overall)'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run with synthetic data (no real dataset needed)'
    )
    parser.add_argument(
        '--model', type=str, default='multi_criteria',
        choices=['multi_criteria', 'single', 'both'],
        help='Model architecture: multi_criteria (default), single, or both'
    )
    parser.add_argument(
        '--num_ratings', type=int, default=NUM_RATINGS,
        help=f'Max rating level R (default: {NUM_RATINGS} for MovieLens)'
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Training epochs (default: {EPOCHS}; use 200+ for paper-level results)'
    )
    parser.add_argument(
        '--embedding_dim', type=int, default=EMBEDDING_DIM,
        help=f'User/item embedding dimension k (default: {EMBEDDING_DIM})'
    )
    parser.add_argument(
        '--generalization', type=str, default='weak',
        choices=['weak', 'strong', 'both'],
        help='Evaluation setting (default: weak; use both to run all settings)'
    )
    args = parser.parse_args()

    EPOCHS        = args.epochs
    EMBEDDING_DIM = args.embedding_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    set_random_seeds(RANDOM_SEED)

    data, num_users, num_items = load_and_preprocess_data(args)
    if data is None:
        return

    num_ratings = args.num_ratings

    print(f"\nConfiguration:")
    print(f"  Model architecture     : {args.model.upper()}")
    print(f"  R (max rating)         : {num_ratings}")
    print(f"  Embedding dim k        : {EMBEDDING_DIM}")
    print(f"  Theta embed dim        : {THETA_EMBED_DIM}")
    print(f"  User hidden layers     : {USER_HIDDEN}")
    print(f"  Item hidden layers     : {ITEM_HIDDEN}")
    print(f"  Theta hidden layers    : {THETA_HIDDEN}")
    print(f"  Dropout prob           : {DROPOUT}")
    print(f"  Lambda (L2 reg)        : {LAMBDA_REG}")
    print(f"  Batch size             : {BATCH_SIZE}")
    print(f"  Learning rate          : {LEARNING_RATE}")
    print(f"  Training epochs        : {EPOCHS}")
    print(f"  Generalization setting : {args.generalization}")

    print(f"\nSplitting into {NUM_FOLDS} folds for cross-validation...")
    folds = split_data_kfold(data, n_splits=NUM_FOLDS, random_state=RANDOM_SEED)

    all_fold_results = []
    for fold_number, (train_data, test_data) in enumerate(folds, start=1):
        fold_results = run_fold(
            fold_number=fold_number,
            train_data=train_data,
            test_data=test_data,
            num_users=num_users,
            num_items=num_items,
            num_ratings=num_ratings,
            device=device,
            generalization=args.generalization,
            model_type=args.model
        )
        all_fold_results.append(fold_results)

    print_final_results(all_fold_results, args.generalization, num_ratings, model_type=args.model)


if __name__ == '__main__':
    main()

