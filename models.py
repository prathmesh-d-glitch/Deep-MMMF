"""
models.py
---------
Deep Maximum Margin Matrix Factorization (Deep MMMF)
------------------------------------------------------
Based on the paper:
  "Deep Maximum Margin Matrix Factorization"
  by Shailendra Kumar, Venkateswara Rao Kagita, Vikas Kumar, Gaurav Niranjan
  Scientific Reports (2026) 16:14518

This file implements the Deep MMMF model which extends the classical
Maximum Margin Matrix Factorization (MMMF) with nonlinear deep neural
transformations on user, item, and threshold embeddings.

Architecture Overview:
-----------------------
  Classical MMMF:
    - User latent factor U_i:  row i of matrix U in R^{n x k}
    - Item latent factor V_j:  row j of matrix V in R^{m x k}
    - Threshold params Theta_{i,r}: (R-1) thresholds per user
    - Prediction score: s_{ij} = U_i . V_j^T
    - Rating predicted by: Theta_{i,r-1} < s_{ij} <= Theta_{i,r}

  Deep MMMF (this file):
    - Same embeddings U, V, Theta as MMMF (same dimensionality k)
    - Applies nonlinear neural transformations Psi_1, Psi_2, Psi_3 to get:
        U_tilde_i = Psi_1(U_i)   -- transformed user embedding
        V_tilde_j = Psi_2(V_j)   -- transformed item embedding
        Theta_tilde_ir = Psi_3(Theta_ir)  -- transformed threshold
    - New prediction score: s_tilde_{ij} = U_tilde_i . V_tilde_j^T
    - Rating predicted by: Theta_tilde_{i,r-1} < s_tilde_{ij} <= Theta_tilde_{i,r}

  Key insight: nonlinearity comes from functional transforms, NOT from
  increasing embedding dimensionality (fair comparison with MMMF).

Loss Function (Eq. 3 in paper):
---------------------------------
  J(W) = Sum_{(i,j) in Omega} Sum_{r=1}^{R-1} h(T^r_{ij} * (Theta_tilde_{i,r} - U_tilde_i . V_tilde_j^T))
         + (lambda/2) * ||W||^2_F

  where the smooth hinge loss h(z) is:
    h(z) = 0              if z >= 1      (correct with margin)
           0.5*(1-z)^2   if 0 < z < 1  (in margin zone)
           0.5 - z       if z <= 0      (wrong side)

  and the indicator T^r_{ij} is:
    T^r_{ij} = +1  if r >= Y_{ij}  (threshold is above or at true rating)
             = -1  if r <  Y_{ij}  (threshold is below true rating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  SMOOTH HINGE LOSS FUNCTION  (Equation 1 from the paper)
# ============================================================

def smooth_hinge_loss(z):
    """
    Compute the smooth hinge loss h(z) as defined in the paper (Eq. 1).

    The smooth hinge loss is a differentiable relaxation of the regular hinge loss:
        h(z) = 0              if z >= 1         (margin satisfied, no loss)
               0.5*(1-z)^2   if 0 < z < 1      (margin partially satisfied)
               0.5 - z       if z <= 0          (constraint violated)

    This function is used to enforce margin-based ordinal constraints.
    It is smooth everywhere (unlike regular hinge loss), which enables
    stable gradient-based optimization.

    Parameters:
        z (Tensor): Input values. These are T^r_{ij} * (threshold - score).

    Returns:
        Tensor: Element-wise smooth hinge loss values (same shape as z).
    """
    # Region 1: z >= 1 -> loss = 0 (constraint satisfied with margin)
    loss_ge1  = torch.zeros_like(z)

    # Region 2: 0 < z < 1 -> loss = 0.5 * (1 - z)^2  (partially satisfied)
    loss_mid  = 0.5 * (1.0 - z) ** 2

    # Region 3: z <= 0 -> loss = 0.5 - z  (constraint violated, linear penalty)
    loss_le0  = 0.5 - z

    # Combine regions using torch.where for element-wise selection:
    # First: pick between region 2 and region 3 (split at z=0)
    loss = torch.where(z > 0, loss_mid, loss_le0)

    # Then: override with 0 wherever z >= 1 (no loss in satisfied region)
    loss = torch.where(z >= 1.0, loss_ge1, loss)

    return loss


# ============================================================
#  DEEP TRANSFORMATION NETWORK  (Psi_1, Psi_2, Psi_3 in the paper)
# ============================================================

class DeepTransformNet(nn.Module):
    """
    A small Multi-Layer Perceptron (MLP) that applies a nonlinear transformation
    to a fixed-dimensional embedding vector.

    This represents the Psi functions from the paper (Section "Deep MMMF"):
        Psi_1: U_i    -> U_tilde_i   (user embedding transform)
        Psi_2: V_j    -> V_tilde_j   (item embedding transform)
        Psi_3: Theta_ir -> Theta_tilde_ir  (threshold embedding transform)

    Architecture:
        input (dim d) -> Linear -> LeakyReLU -> Dropout
                      -> Linear -> LeakyReLU -> Dropout
                      -> Linear -> output (dim d)

    The input and output dimensionality are kept IDENTICAL to preserve
    the original embedding size k (this is the key design choice that
    isolates nonlinearity from dimensional expansion).

    Layer sizes follow Table 2 in the paper:
        For user/item:  256 -> 128 -> k  (where k = embedding_dim)
        For threshold:  16  -> 4   -> k
    """

    def __init__(self, input_dim, hidden_sizes, dropout_prob=0.2):
        """
        Initialize the deep transformation network.

        Parameters:
            input_dim    (int)       : Input (and output) dimensionality = embedding size.
            hidden_sizes (list[int]) : List of hidden layer sizes. E.g., [256, 128] or [16, 4].
            dropout_prob (float)     : Dropout probability between layers (paper uses 0.2 or 0.3).
        """
        super(DeepTransformNet, self).__init__()

        # Build the sequential layers dynamically based on hidden_sizes list
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            # Linear transformation: projects from prev_dim -> hidden_dim
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # LeakyReLU activation: f(x) = x if x > 0, else alpha*x  (alpha=0.01 default)
            # Paper uses LeakyReLU throughout (Table 2).
            layers.append(nn.LeakyReLU(negative_slope=0.01))

            # Dropout: randomly zeroes neurons to prevent overfitting.
            # Paper applies dropout between layers with p=0.2 or 0.3 (Table 2).
            layers.append(nn.Dropout(p=dropout_prob))

            prev_dim = hidden_dim

        # Final projection back to input_dim (preserves embedding dimensionality as per paper)
        layers.append(nn.Linear(prev_dim, input_dim))

        # Wrap all layers in a single sequential module for clean forward pass
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply the nonlinear transformation to the input embedding.

        Uses a RESIDUAL CONNECTION (He et al., ResNet, CVPR 2016):
            output = Psi(x) + x

        The residual path ensures gradients flow directly back to the raw
        embedding during backprop, preventing vanishing gradients and
        enabling the transform to learn only the *correction* delta on
        top of the identity mapping. This consistently improves convergence
        and final accuracy versus a plain MLP.

        Parameters:
            x (Tensor): Shape (..., input_dim) -- the raw embedding vector.

        Returns:
            Tensor: Shape (..., input_dim) -- the transformed embedding Psi(x) + x.
        """
        return self.net(x) + x   # residual: learn delta, not full mapping


# ============================================================
#  DEEP MMMF MODEL  (Algorithm 1 in the paper)
# ============================================================

class DeepMMMF(nn.Module):
    """
    Deep Maximum Margin Matrix Factorization (Deep MMMF).

    Extends classical MMMF by applying learnable nonlinear transformations
    (via small MLPs Psi_1, Psi_2, Psi_3) to user, item, and threshold embeddings.

    Key Components:
    ---------------
    1. Embedding Layers (same as MMMF, same dimensionality):
         - user_embedding  : Lookup table W_U in R^{n x k}  (U in the paper)
         - item_embedding  : Lookup table W_V in R^{m x k}  (V in the paper)
         - threshold_embedding: Per-user threshold embeddings (R-1 per user)

    2. Deep Transformation Networks (the NEW component vs MMMF):
         - user_transform      : Psi_1 -- transforms U_i -> U_tilde_i
         - item_transform      : Psi_2 -- transforms V_j -> V_tilde_j
         - threshold_transform : Psi_3 -- transforms Theta_{i,r} -> Theta_tilde_{i,r}

    3. Prediction (forward pass):
         - Compute U_tilde_i . V_tilde_j^T = score_{ij}
         - Rating r_hat_{ij} = r such that Theta_tilde_{i,r-1} < score_{ij} <= Theta_tilde_{i,r}

    Hyperparameters (from Table 2 in the paper for MovieLens weak generalization):
    -------------------------------------------------------------------------------
        embedding_dim    = 100   (k in the paper, user/item embed size)
        theta_embed_dim  = 32    (theta embedding size, Table 2 column "Embedding size")
        user_hidden      = [256, 128]   (user network layers)
        item_hidden      = [256, 128]   (item network layers)
        theta_hidden     = [16, 4]      (theta hidden layers: "16->4" in Table 2)
        dropout          = 0.2
        lambda_reg       = 1e-6
    """

    def __init__(self, num_users, num_items, num_ratings,
                 embedding_dim=100,
                 theta_embed_dim=32,
                 user_hidden_sizes=None,
                 item_hidden_sizes=None,
                 theta_hidden_sizes=None,
                 dropout_prob=0.2,
                 bpr_alpha=0.5):
        """
        Initialize all components of Deep MMMF.

        Parameters:
            num_users       (int)   : Total number of unique users (n).
            num_items       (int)   : Total number of unique items (m).
            num_ratings     (int)   : Maximum rating level R (e.g., 5 for {1,2,3,4,5}).
            embedding_dim   (int)   : Latent dimensionality k for user and item (paper: 100).
            theta_embed_dim (int)   : Dimensionality of threshold embedding vector (paper: 32).
            user_hidden_sizes(list) : Hidden layer sizes for Psi_1 (default: [256, 128]).
            item_hidden_sizes(list) : Hidden layer sizes for Psi_2 (default: [256, 128]).
            theta_hidden_sizes(list): Hidden layer sizes for Psi_3 (default: [16, 4]).
            dropout_prob    (float) : Dropout rate (paper: 0.2 for MovieLens, 0.3 for EachMovie).
            bpr_alpha       (float) : Weight of BPR ranking loss vs hinge loss (default: 0.5).
                                      Higher values prioritise ranking over rating accuracy.
                                      (BPR: Rendle et al., "Bayesian Personalized Ranking", UAI 2009)
        """
        super(DeepMMMF, self).__init__()

        # Store configuration for use in forward pass and loss computation
        self.num_users      = num_users
        self.num_items      = num_items
        self.num_ratings    = num_ratings        # R (maximum rating)
        self.embedding_dim  = embedding_dim      # k
        self.theta_embed_dim = theta_embed_dim   # d_theta
        self.num_thresholds = num_ratings - 1    # R-1 thresholds per user
        self.bpr_alpha      = bpr_alpha          # BPR ranking loss weight

        # Use default hidden sizes matching Table 2 (MovieLens config) if not provided
        if user_hidden_sizes is None:
            user_hidden_sizes = [256, 128]
        if item_hidden_sizes is None:
            item_hidden_sizes = [256, 128]
        if theta_hidden_sizes is None:
            theta_hidden_sizes = [16, 4]

        # ----------------------------------------------------------------
        #  EMBEDDING LAYERS
        #  These are the W_U, W_V, W_Theta weight matrices from the paper.
        #  Each user/item one-hot vector is multiplied with these matrices
        #  to get dense latent representations U_i, V_j.
        # ----------------------------------------------------------------

        # User embedding: one-hot(user_i) x W_U -> U_i in R^k
        # nn.Embedding is a trainable lookup table (weight matrix) of size (n, k)
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

        # Item embedding: one-hot(item_j) x W_V -> V_j in R^k
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        # Threshold embedding:
        # Classical MMMF has scalar thresholds Theta_{i,r} for each user i and level r.
        # Deep MMMF embeds each threshold as a vector of size theta_embed_dim,
        # then transforms and projects it to a scalar.
        # Total entries in table: num_users * (R-1)
        # (User i, threshold level r) is at index: i*(R-1) + (r-1)
        self.threshold_embedding = nn.Embedding(
            num_embeddings=num_users * self.num_thresholds,
            embedding_dim=theta_embed_dim
        )

        # Projection layer: maps transformed threshold vector -> scalar threshold value
        # This gives one scalar Theta_tilde_{i,r} per (user, rating level) pair
        self.threshold_projection = nn.Linear(theta_embed_dim, 1)

        # Dropout applied after embedding lookup (Table 2: dropout=0.2 on embeddings)
        self.embed_dropout = nn.Dropout(p=dropout_prob)

        # ----------------------------------------------------------------
        #  DEEP TRANSFORMATION NETWORKS  (Psi_1, Psi_2, Psi_3)
        #  These are the NEW components that distinguish Deep MMMF from MMMF.
        #  They introduce nonlinearity without changing embedding dimensions.
        # ----------------------------------------------------------------

        # Psi_1: User transformation network
        # U_i in R^k  ->  [256 -> 128 -> k]  -> U_tilde_i in R^k
        self.user_transform = DeepTransformNet(
            input_dim=embedding_dim,
            hidden_sizes=user_hidden_sizes,
            dropout_prob=dropout_prob
        )

        # Psi_2: Item transformation network
        # V_j in R^k  ->  [256 -> 128 -> k]  -> V_tilde_j in R^k
        self.item_transform = DeepTransformNet(
            input_dim=embedding_dim,
            hidden_sizes=item_hidden_sizes,
            dropout_prob=dropout_prob
        )

        # Psi_3: Threshold transformation network
        # Theta_{i,r} in R^{d_theta}  ->  [16 -> 4 -> d_theta]  -> transformed threshold
        self.threshold_transform = DeepTransformNet(
            input_dim=theta_embed_dim,
            hidden_sizes=theta_hidden_sizes,
            dropout_prob=dropout_prob
        )

        # ----------------------------------------------------------------
        #  WEIGHT INITIALIZATION
        # ----------------------------------------------------------------
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model parameters.

        - Embeddings: small random values from N(0, 0.01) to start near zero
        - Linear layers in transform nets: Xavier uniform (good for LeakyReLU)
        - Biases: initialized to zero
        """
        # Small normal initialization for embeddings (avoids large initial scores)
        nn.init.normal_(self.user_embedding.weight,      mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight,      mean=0.0, std=0.01)
        nn.init.normal_(self.threshold_embedding.weight, mean=0.0, std=0.01)

        # Xavier uniform for all Linear layers in transform networks
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_threshold_indices(self, user_ids, device):
        """
        Compute flat indices into the threshold embedding table for each
        (user, rating_level) pair in the batch.

        The threshold embedding table has shape [num_users * (R-1), d_theta].
        For user i and threshold level index t (0-indexed, 0 to R-2):
            flat_index = i * (R-1) + t

        Parameters:
            user_ids (Tensor): Shape (batch_size,) -- user indices.
            device   : torch.device

        Returns:
            Tensor: Shape (batch_size, R-1) -- flat indices into threshold table.
        """
        # Base index for each user: user_i * (R-1)
        # Shape: (batch_size, 1) for broadcasting
        base_indices = (user_ids * self.num_thresholds).unsqueeze(1)

        # Threshold level offsets: 0, 1, ..., R-2
        # Shape: (1, R-1)
        threshold_offsets = torch.arange(self.num_thresholds, device=device).unsqueeze(0)

        # Broadcast addition: (batch_size, 1) + (1, R-1) -> (batch_size, R-1)
        indices = base_indices + threshold_offsets

        return indices

    def get_transformed_embeddings(self, user_ids, item_ids):
        """
        Compute the transformed user, item, and threshold embeddings.

        This implements Algorithm 1 (Steps 1-4) from the paper:
          Step 1: Look up raw embeddings U_i, V_j, Theta_{i,r}
          Step 2: Apply dropout to embeddings
          Step 3: Pass through Psi_1, Psi_2, Psi_3 to get transformed versions
          Step 4: Project threshold vectors to scalar values

        Parameters:
            user_ids (Tensor): Shape (batch_size,)
            item_ids (Tensor): Shape (batch_size,)

        Returns:
            user_transformed  (Tensor): Shape (batch_size, k) -- U_tilde_i
            item_transformed  (Tensor): Shape (batch_size, k) -- V_tilde_j
            thresholds_scalar (Tensor): Shape (batch_size, R-1) -- Theta_tilde_{i,1..R-1}
        """
        device = user_ids.device
        batch_size = user_ids.shape[0]

        # ------ Step 1: Lookup raw embeddings from embedding tables ------

        # User embedding: U_i in R^k
        user_embed = self.user_embedding(user_ids)   # Shape: (batch, k)

        # Item embedding: V_j in R^k
        item_embed = self.item_embedding(item_ids)   # Shape: (batch, k)

        # Threshold embeddings: Theta_{i,r} in R^{d_theta} for each threshold level r
        # First compute flat indices for all (user, threshold level) pairs
        threshold_indices = self._get_threshold_indices(user_ids, device)  # (batch, R-1)

        # Flatten to 1D for embedding lookup, then reshape back to (batch, R-1, d_theta)
        threshold_embed = self.threshold_embedding(
            threshold_indices.view(-1)   # Shape: (batch*(R-1),)
        ).view(batch_size, self.num_thresholds, -1)
        # Final shape: (batch, R-1, d_theta)

        # ------ Step 2: Apply embedding dropout ------
        user_embed      = self.embed_dropout(user_embed)      # (batch, k)
        item_embed      = self.embed_dropout(item_embed)      # (batch, k)
        threshold_embed = self.embed_dropout(threshold_embed) # (batch, R-1, d_theta)

        # ------ Step 3: Apply deep nonlinear transformations ------

        # Psi_1(U_i): transform user embedding through the user MLP
        # Shape: (batch, k) -> user_transform -> (batch, k)
        user_transformed = self.user_transform(user_embed)

        # Psi_2(V_j): transform item embedding through the item MLP
        # Shape: (batch, k) -> item_transform -> (batch, k)
        item_transformed = self.item_transform(item_embed)

        # Psi_3(Theta_{i,r}): transform threshold embeddings through threshold MLP
        # Reshape from (batch, R-1, d_theta) to (batch*(R-1), d_theta) for MLP
        threshold_flat = threshold_embed.view(batch_size * self.num_thresholds, -1)
        threshold_transformed = self.threshold_transform(threshold_flat)
        # Reshape back to (batch, R-1, d_theta)
        threshold_transformed = threshold_transformed.view(batch_size, self.num_thresholds, -1)

        # ------ Step 4: Project threshold vectors to scalar values ------
        # threshold_projection maps R^{d_theta} -> R^1 (a scalar per threshold)
        # Shape: (batch, R-1, d_theta) -> (batch, R-1, 1) -> squeeze -> (batch, R-1)
        thresholds_scalar = self.threshold_projection(threshold_transformed).squeeze(-1)

        return user_transformed, item_transformed, thresholds_scalar

    def compute_bpr_loss(self, user_trans, pos_scores, item_ids, true_ratings):
        """
        Bayesian Personalized Ranking (BPR) loss for explicit feedback.

        Source: Rendle et al., "BPR: Bayesian Personalized Ranking from
        Implicit Feedback", UAI 2009.

        Adapted for EXPLICIT (rated) feedback:
          - For each observed (user, pos_item) in the batch, sample a
            random negative item from the full item space.
          - Encourage: score(user, pos_item) > score(user, neg_item)
          - Weight each pair by the true rating so that high-rated items
            are pushed further above negatives than low-rated items.

        This directly optimises the pairwise ranking objective that
        underlies F1, F2, MAP, MRR, and FCP metrics.

        Parameters:
            user_trans   (Tensor): Shape (batch, k) -- transformed user embeddings.
            pos_scores   (Tensor): Shape (batch,)   -- interaction scores of observed pairs.
            item_ids     (Tensor): Shape (batch,)   -- positive item indices.
            true_ratings (Tensor): Shape (batch,)   -- ground-truth ratings (1-indexed).

        Returns:
            Tensor: Scalar BPR loss value summed across the batch.
        """
        device = item_ids.device

        # Sample random negative items
        neg_item_ids = torch.randint(0, self.num_items, item_ids.shape, device=device)
        neg_embed    = self.embed_dropout(self.item_embedding(neg_item_ids))  # (batch, k)
        neg_trans    = self.item_transform(neg_embed)                         # (batch, k)
        neg_scores   = (user_trans * neg_trans).sum(dim=1)                    # (batch,)

        # Rating-weighted BPR loss
        # Items with higher ratings should be pushed further above negatives.
        rating_weights = true_ratings.float() / self.num_ratings

        # log-sigmoid of score difference: minimized when pos_score >> neg_scores
        bpr = -F.logsigmoid(pos_scores - neg_scores)   # (batch,)
        return (rating_weights * bpr).sum()

    def compute_loss(self, user_ids, item_ids, true_ratings, lambda_reg=1e-6):
        """
        Compute the Deep MMMF training loss (Equation 3 in the paper).

        Full loss:
          J(W) = Sum_{(i,j) in Omega} Sum_{r=1}^{R-1}
                   h(T^r_{ij} * (Theta_tilde_{i,r} - U_tilde_i . V_tilde_j^T))
                 + (lambda/2) * ||W||^2_F

        The loss enforces ordinal constraints: for a true rating Y_{ij},
        the score should satisfy Theta_{i,Y-1} < score <= Theta_{i,Y}.

        Parameters:
            user_ids     (Tensor): Shape (batch_size,) -- user indices
            item_ids     (Tensor): Shape (batch_size,) -- item indices
            true_ratings (Tensor): Shape (batch_size,) -- ground truth ratings (1-indexed)
            lambda_reg   (float) : L2 regularization strength (paper: 1e-6 for MovieLens)

        Returns:
            loss (Tensor): Scalar training loss value.
        """
        # Step 1: Get transformed embeddings from all three paths
        user_trans, item_trans, thresholds = self.get_transformed_embeddings(user_ids, item_ids)
        # user_trans:  (batch, k)
        # item_trans:  (batch, k)
        # thresholds:  (batch, R-1)  -- scalar threshold value per level

        # Step 2: Compute interaction scores s_tilde_{ij} = U_tilde_i . V_tilde_j^T
        # Element-wise product summed over k (equivalent to dot product per sample)
        scores = (user_trans * item_trans).sum(dim=1)  # Shape: (batch,)

        # Step 3: Expand scores to broadcast against all R-1 thresholds
        # Shape: (batch,) -> (batch, 1) -> (batch, R-1)
        scores_expanded = scores.unsqueeze(1).expand_as(thresholds)

        # Step 4: Compute T^r_{ij} indicator values (Equation 2 in the paper)
        # For each sample and threshold level r (1-indexed):
        #   T^r_{ij} = +1  if r >= Y_{ij}   (threshold at or above true rating)
        #            = -1  if r <  Y_{ij}   (threshold below true rating)
        #
        # r_levels: tensor [1, 2, ..., R-1], broadcast to (batch, R-1)
        r_levels = torch.arange(1, self.num_ratings,
                                device=user_ids.device).float().unsqueeze(0)  # (1, R-1)

        # Expand true ratings from (batch,) to (batch, R-1) for broadcasting
        true_expanded = true_ratings.float().unsqueeze(1).expand_as(thresholds)

        # T: +1 where r >= Y_{ij}, -1 where r < Y_{ij}
        T = torch.where(r_levels >= true_expanded,
                        torch.ones_like(thresholds),
                        -torch.ones_like(thresholds))
        # Shape: (batch, R-1)

        # Step 5: Compute the argument to the smooth hinge loss
        # z^r_{ij} = T^r_{ij} * (Theta_tilde_{i,r} - s_tilde_{ij})
        z = T * (thresholds - scores_expanded)  # Shape: (batch, R-1)

        # Step 6: Apply smooth hinge loss element-wise, then sum (Eq. 1, Eq. 3)
        hinge = smooth_hinge_loss(z)  # Shape: (batch, R-1)
        data_loss = hinge.sum()       # Scalar

        # Step 7: L2 regularization on all trainable parameters (Frobenius norm)
        # (lambda/2) * ||W||^2_F
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss = l2_loss + (param ** 2).sum()
        l2_loss = (lambda_reg / 2.0) * l2_loss

        # Step 8: BPR ranking loss (Rendle et al., UAI 2009)
        # Encourages the model to rank relevant items above random items,
        # directly optimising AUC / ranking metrics (F1, MAP, MRR, FCP).
        bpr_loss = self.compute_bpr_loss(user_trans, scores, item_ids, true_ratings)

        # Total loss = hinge (rating accuracy) + BPR (ranking) + L2 (regularisation)
        return data_loss + self.bpr_alpha * bpr_loss + l2_loss

    def predict_ratings(self, user_ids, item_ids):
        """
        Predict ordinal ratings for a batch of (user, item) pairs.

        Prediction rule from the paper (after Eq. 3):
            r_hat_{ij} = r  such that  Theta_tilde_{i,r-1} < s_tilde_{ij} <= Theta_tilde_{i,r}

        Boundary conditions (paper sets them to -inf and +inf):
            Theta_tilde_{i,0} = -inf
            Theta_tilde_{i,R} = +inf

        Implementation: count how many thresholds the score exceeds.
        If score > Theta_{i,r} for k thresholds, then predicted rating = k+1.

        Parameters:
            user_ids (Tensor): Shape (N,) -- user indices
            item_ids (Tensor): Shape (N,) -- item indices

        Returns:
            Tensor: Shape (N,) -- predicted integer ratings in {1, ..., R}
        """
        with torch.no_grad():
            # Get transformed embeddings in inference mode
            user_trans, item_trans, thresholds = self.get_transformed_embeddings(user_ids, item_ids)

            # Compute interaction scores
            scores = (user_trans * item_trans).sum(dim=1)  # (N,)

            # Expand scores for comparison with all R-1 thresholds
            scores_expanded = scores.unsqueeze(1).expand_as(thresholds)  # (N, R-1)

            # Count how many thresholds the score exceeds
            # score > Theta_{i,r} means rating is at least r+1
            # So predicted rating = (number of thresholds exceeded) + 1
            exceeds = (scores_expanded > thresholds).sum(dim=1)  # (N,) in {0, ..., R-1}

            # Add 1 (since ratings are 1-indexed) and clip to [1, R]
            predicted = (exceeds + 1).clamp(1, self.num_ratings)

        return predicted

    def predict_scores(self, user_ids, item_ids):
        """
        Return raw continuous interaction scores for a batch of (user, item) pairs.

        Unlike predict_ratings() which returns quantised integers {1,...,R}, this
        method returns the raw dot-product score s_tilde_{ij} = U_tilde_i · V_tilde_j.

        WHY THIS MATTERS FOR RANKING:
        When many items receive the same predicted integer rating (e.g. all "3 stars"),
        their relative ranking is arbitrary — which collapses F1, MAP, MRR, and FCP.
        Continuous scores from the latent space carry fine-grained preference signal
        and properly differentiate items for per-user ranking, dramatically improving
        all ranking metrics.

        Usage:
          - Use predict_ratings() for MAE / NMAE (rating accuracy metrics).
          - Use predict_scores()  for F1, F2, FCP, MAP, MRR (ranking metrics).

        Parameters:
            user_ids (Tensor): Shape (N,) -- user indices
            item_ids (Tensor): Shape (N,) -- item indices

        Returns:
            Tensor: Shape (N,) -- raw float interaction scores (higher = more preferred)
        """
        with torch.no_grad():
            user_trans, item_trans, _ = self.get_transformed_embeddings(user_ids, item_ids)
            scores = (user_trans * item_trans).sum(dim=1)   # (N,) continuous scores
        return scores

    def forward(self, user_ids, item_ids):
        """
        Forward pass: returns predicted ratings (integer values in {1, ..., R}).

        This is the inference-time method (used for evaluation/testing).
        For training, call compute_loss() directly to get the hinge loss.

        Parameters:
            user_ids (Tensor): Shape (batch_size,) -- user indices
            item_ids (Tensor): Shape (batch_size,) -- item indices

        Returns:
            Tensor: Shape (batch_size,) -- predicted ratings as floats
        """
        # predict_ratings returns integer predictions, convert to float for metric computation
        return self.predict_ratings(user_ids, item_ids).float()


# ============================================================
#  CRITERIA RANKING AGGREGATOR
# ============================================================

class CriteriaRankingAggregator(nn.Module):
    """
    Ranking Aggregator Network for Multi-Criteria Deep MMMF.

    Takes transformed criterion vectors from each of the 7 independently factorized
    criteria matrices, applies dynamic attention-based criterion weighting, and passes
    the combined representation through a multi-layer ranking aggregator to compute
    a continuous overall interaction score for ranking.

    Architecture:
    -------------
    1. Multi-Criteria Attention:
       Computes dynamic criterion importance weights alpha_{ij}^(c) based on the
       latent criterion interaction vectors.
    2. Context Representation:
       Context vector c_{ij} = Sum_{c=1}^C alpha_{ij}^(c) * z_{ij}^(c)
    3. Multi-Layer Ranking MLP:
       Concatenates [c_{ij}, z_{ij}^{(1..C)}, s_{ij}^{(1..C)}] and processes through
       residual/LeakyReLU layers with dropout to output scalar ranking score S_hat_{ij}.
    """

    def __init__(self, num_criteria=7, criterion_vector_dim=32,
                 hidden_sizes=None, dropout_prob=0.2):
        super(CriteriaRankingAggregator, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.num_criteria = num_criteria
        self.criterion_vector_dim = criterion_vector_dim

        # Dynamic attention network over criteria
        self.attn_net = nn.Sequential(
            nn.Linear(criterion_vector_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False)
        )

        # Total input dimension to aggregator MLP:
        # - Attended context vector: criterion_vector_dim
        # - Concatenated criteria vectors: num_criteria * criterion_vector_dim
        # - Direct criterion interaction scores: num_criteria
        in_dim = criterion_vector_dim + (num_criteria * criterion_vector_dim) + num_criteria

        layers = []
        prev_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.aggregator = nn.Sequential(*layers)

    def forward(self, criterion_vectors, criterion_scores):
        """
        Parameters:
            criterion_vectors (Tensor): Shape (batch, num_criteria, criterion_vector_dim)
            criterion_scores  (Tensor): Shape (batch, num_criteria)

        Returns:
            overall_score (Tensor): Shape (batch,) continuous ranking score S_hat_{ij}
            attn_weights  (Tensor): Shape (batch, num_criteria) dynamic attention weights
        """
        batch_size = criterion_vectors.shape[0]

        # 1. Compute dynamic attention weights across the 7 criteria
        attn_logits = self.attn_net(criterion_vectors).squeeze(-1)  # (batch, num_criteria)
        attn_weights = F.softmax(attn_logits, dim=-1)               # (batch, num_criteria)

        # 2. Context vector via weighted attention pooling
        context = torch.sum(criterion_vectors * attn_weights.unsqueeze(-1), dim=1)  # (batch, criterion_vector_dim)

        # 3. Flattened criteria representations
        flat_vectors = criterion_vectors.view(batch_size, -1)       # (batch, num_criteria * criterion_vector_dim)

        # 4. Joint feature representation
        features = torch.cat([context, flat_vectors, criterion_scores], dim=-1)

        # 5. Output continuous ranking score
        overall_score = self.aggregator(features).squeeze(-1)       # (batch,)
        return overall_score, attn_weights


# ============================================================
#  MULTI-CRITERIA DEEP MMMF
# ============================================================

class MultiCriteriaDeepMMMF(nn.Module):
    """
    Multi-Criteria Deep Maximum Margin Matrix Factorization (MC-Deep MMMF).

    Extends Deep MMMF to simultaneously factorize multiple criteria matrices
    (e.g., TripAdvisor's 7 criteria: Value, Rooms, Location, Cleanliness, Check-in,
    Service, Business Service) with coupled threshold embeddings, passing transformed
    criterion vectors to a final ranking aggregator.

    Key Architectural Innovations:
    ------------------------------
    1. Independent Factorization per Criterion:
       Each criterion c has independent user embeddings U^(c) in R^{n x k}, item
       embeddings V^(c) in R^{m x k}, and nonlinear deep transforms Psi_1^(c), Psi_2^(c).
       Each criterion maintains its own latent geometry, isolating aspect-specific
       preferences without dimensional interference.

    2. Coupled Threshold Embeddings:
       User grading severity is correlated across criteria (harsh vs generous raters),
       yet individual criteria exhibit distinct standard shifts. We couple thresholds:
           Theta_{i, r}^(c) = E_{i, r}^(base) + C_{c, r}^(crit)
       where:
         - E_{i, r}^(base) is a shared user threshold embedding across all criteria and overall rating.
         - C_{c, r}^(crit) is a criterion-specific threshold calibration embedding.
       Coupled threshold representations are transformed through Psi_3 and projected
       to scalar thresholds Theta_tilde_{i, r}^(c) and Theta_tilde_{i, r}^(overall).

    3. Transformed Criterion Vectors & Ranking Aggregator:
       For each criterion, transformed user and item factors produce an interaction
       vector z_{ij}^(c) combining element-wise interaction and scalar score.
       The 7 transformed criterion vectors are passed to a neural Ranking Aggregator
       with dynamic attention, yielding continuous ranking score S_hat_{ij}.

    4. Joint Multi-Task Ordinal & Ranking Optimization:
       Simultaneously minimizes:
         - Smooth hinge losses for each of the 7 criteria matrices
         - Smooth hinge loss for the overall rating
         - BPR pairwise ranking loss on the aggregated score S_hat_{ij}
         - Frobenius L2 weight regularization
    """

    def __init__(self, num_users, num_items, num_ratings=5,
                 criteria_names=None,
                 embedding_dim=64,
                 theta_embed_dim=32,
                 criterion_vector_dim=32,
                 user_hidden_sizes=None,
                 item_hidden_sizes=None,
                 theta_hidden_sizes=None,
                 aggregator_hidden_sizes=None,
                 dropout_prob=0.2,
                 bpr_alpha=0.5,
                 criteria_alpha=1.0):
        super(MultiCriteriaDeepMMMF, self).__init__()

        if criteria_names is None:
            criteria_names = [
                'value', 'rooms', 'location', 'cleanliness',
                'checkin', 'service', 'business_service'
            ]

        self.num_users            = num_users
        self.num_items            = num_items
        self.num_ratings          = num_ratings
        self.num_thresholds       = num_ratings - 1
        self.criteria_names       = criteria_names
        self.num_criteria         = len(criteria_names)
        self.embedding_dim        = embedding_dim
        self.theta_embed_dim      = theta_embed_dim
        self.criterion_vector_dim = criterion_vector_dim
        self.dropout_prob         = dropout_prob
        self.bpr_alpha            = bpr_alpha
        self.criteria_alpha       = criteria_alpha

        if user_hidden_sizes is None:
            user_hidden_sizes = [128, 64]
        if item_hidden_sizes is None:
            item_hidden_sizes = [128, 64]
        if theta_hidden_sizes is None:
            theta_hidden_sizes = [16, 4]
        if aggregator_hidden_sizes is None:
            aggregator_hidden_sizes = [128, 64]

        # ------------------------------------------------------------
        # 1. INDEPENDENT FACTORIZATION PATHS FOR EACH CRITERION
        # ------------------------------------------------------------
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(num_users, embedding_dim) for _ in range(self.num_criteria)
        ])
        self.item_embeddings = nn.ModuleList([
            nn.Embedding(num_items, embedding_dim) for _ in range(self.num_criteria)
        ])

        self.user_transforms = nn.ModuleList([
            DeepTransformNet(embedding_dim, user_hidden_sizes, dropout_prob)
            for _ in range(self.num_criteria)
        ])
        self.item_transforms = nn.ModuleList([
            DeepTransformNet(embedding_dim, item_hidden_sizes, dropout_prob)
            for _ in range(self.num_criteria)
        ])

        # Criterion feature projector: maps (U_tilde_i * V_tilde_j, s_{ij}) -> z_{ij}^(c)
        self.criterion_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim + 1, criterion_vector_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=dropout_prob)
            )
            for _ in range(self.num_criteria)
        ])

        # ------------------------------------------------------------
        # 2. COUPLED THRESHOLD EMBEDDINGS
        # ------------------------------------------------------------
        # Shared user threshold base embedding: captures user's rating stringency
        # Table size: num_users * (R - 1)
        self.shared_user_threshold_embedding = nn.Embedding(
            num_embeddings=num_users * self.num_thresholds,
            embedding_dim=theta_embed_dim
        )

        # Criterion-specific threshold offset table:
        # Indices 0 .. num_criteria - 1 for criteria, and index num_criteria for overall rating
        self.criterion_threshold_embedding = nn.Embedding(
            num_embeddings=(self.num_criteria + 1) * self.num_thresholds,
            embedding_dim=theta_embed_dim
        )

        # Nonlinear threshold transformation network Psi_3
        self.threshold_transform = DeepTransformNet(
            input_dim=theta_embed_dim,
            hidden_sizes=theta_hidden_sizes,
            dropout_prob=dropout_prob
        )

        # Criterion-specific scalar threshold projections
        # (num_criteria for criteria + 1 for overall rating)
        self.threshold_projections = nn.ModuleList([
            nn.Linear(theta_embed_dim, 1) for _ in range(self.num_criteria + 1)
        ])

        # Embedding dropout
        self.embed_dropout = nn.Dropout(p=dropout_prob)

        # ------------------------------------------------------------
        # 3. FINAL RANKING AGGREGATOR
        # ------------------------------------------------------------
        self.ranking_aggregator = CriteriaRankingAggregator(
            num_criteria=self.num_criteria,
            criterion_vector_dim=criterion_vector_dim,
            hidden_sizes=aggregator_hidden_sizes,
            dropout_prob=dropout_prob
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all model parameters."""
        for u_emb in self.user_embeddings:
            nn.init.normal_(u_emb.weight, mean=0.0, std=0.01)
        for v_emb in self.item_embeddings:
            nn.init.normal_(v_emb.weight, mean=0.0, std=0.01)

        nn.init.normal_(self.shared_user_threshold_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.criterion_threshold_embedding.weight,   mean=0.0, std=0.01)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_coupled_thresholds(self, user_ids, target_idx):
        """
        Compute coupled scalar thresholds for a given target index.

        target_idx in [0, num_criteria - 1]: for a specific criterion
        target_idx == num_criteria: for the overall rating

        Coupled Threshold Equation:
            Theta_{i, r}^(c) = E_{i, r}^(base) + C_{c, r}^(crit)
            Theta_tilde_{i, r}^(c) = Linear_c(Psi_3(Theta_{i, r}^(c)))

        Parameters:
            user_ids   (Tensor): Shape (batch,)
            target_idx (int)   : Index of criterion or overall rating

        Returns:
            thresholds_scalar (Tensor): Shape (batch, R - 1)
        """
        device = user_ids.device
        batch_size = user_ids.shape[0]

        # 1. Shared base user threshold embedding lookup
        base_indices = (user_ids * self.num_thresholds).unsqueeze(1)
        threshold_offsets = torch.arange(self.num_thresholds, device=device).unsqueeze(0)
        user_thresh_indices = (base_indices + threshold_offsets).view(-1)  # (batch * (R - 1),)

        user_thresh = self.shared_user_threshold_embedding(
            user_thresh_indices
        ).view(batch_size, self.num_thresholds, self.theta_embed_dim)

        # 2. Criterion-specific threshold offset lookup
        crit_offset_base = target_idx * self.num_thresholds
        crit_indices = crit_offset_base + torch.arange(self.num_thresholds, device=device)
        crit_thresh = self.criterion_threshold_embedding(crit_indices).unsqueeze(0)  # (1, R - 1, d_theta)

        # 3. Couple the shared user stringency and criterion-specific offset
        coupled_thresh = user_thresh + crit_thresh
        coupled_thresh = self.embed_dropout(coupled_thresh)

        # 4. Deep nonlinear transformation Psi_3
        coupled_flat = coupled_thresh.view(batch_size * self.num_thresholds, self.theta_embed_dim)
        trans_flat = self.threshold_transform(coupled_flat)
        trans_thresh = trans_flat.view(batch_size, self.num_thresholds, self.theta_embed_dim)

        # 5. Project to scalar thresholds for this criterion/overall
        scalar_thresholds = self.threshold_projections[target_idx](trans_thresh).squeeze(-1)
        return scalar_thresholds  # (batch, R - 1)

    def forward_criteria(self, user_ids, item_ids):
        """
        Forward pass for all 7 independent criteria factorizations.

        Parameters:
            user_ids (Tensor): Shape (batch,)
            item_ids (Tensor): Shape (batch,)

        Returns:
            criterion_vectors (Tensor): Shape (batch, num_criteria, criterion_vector_dim)
            criterion_scores  (Tensor): Shape (batch, num_criteria)
        """
        vectors = []
        scores = []

        for c in range(self.num_criteria):
            u_raw = self.embed_dropout(self.user_embeddings[c](user_ids))
            v_raw = self.embed_dropout(self.item_embeddings[c](item_ids))

            u_trans = self.user_transforms[c](u_raw)
            v_trans = self.item_transforms[c](v_raw)

            # Scalar interaction score for criterion c
            s_c = (u_trans * v_trans).sum(dim=1)  # (batch,)
            scores.append(s_c)

            # Transformed criterion feature vector
            inter = u_trans * v_trans              # (batch, k)
            feat = torch.cat([inter, s_c.unsqueeze(1)], dim=1)  # (batch, k + 1)
            z_c = self.criterion_projectors[c](feat)            # (batch, criterion_vector_dim)
            vectors.append(z_c)

        criterion_vectors = torch.stack(vectors, dim=1)  # (batch, num_criteria, d_c)
        criterion_scores  = torch.stack(scores,  dim=1)  # (batch, num_criteria)
        return criterion_vectors, criterion_scores

    def compute_bpr_loss(self, user_ids, pos_overall_scores, pos_item_ids, true_overall_ratings):
        """
        Bayesian Personalized Ranking (BPR) loss on aggregated ranking scores.

        Parameters:
            user_ids             (Tensor): Shape (batch,)
            pos_overall_scores   (Tensor): Shape (batch,) aggregated score for observed pairs
            pos_item_ids         (Tensor): Shape (batch,)
            true_overall_ratings (Tensor): Shape (batch,)

        Returns:
            Tensor: Scalar BPR loss
        """
        device = user_ids.device
        neg_item_ids = torch.randint(0, self.num_items, pos_item_ids.shape, device=device)

        neg_vectors, neg_scores = self.forward_criteria(user_ids, neg_item_ids)
        neg_overall_scores, _ = self.ranking_aggregator(neg_vectors, neg_scores)

        rating_weights = true_overall_ratings.float() / self.num_ratings
        bpr = -F.logsigmoid(pos_overall_scores - neg_overall_scores)
        return (rating_weights * bpr).sum()

    def compute_loss(self, user_ids, item_ids, criteria_ratings, overall_ratings, lambda_reg=1e-6):
        """
        Compute joint multi-task loss for Multi-Criteria Deep MMMF:
          Loss = Overall_Hinge + criteria_alpha * (1/C * Sum Criteria_Hinge) + bpr_alpha * BPR + L2_reg

        Parameters:
            user_ids         (Tensor): Shape (batch,)
            item_ids         (Tensor): Shape (batch,)
            criteria_ratings (Tensor): Shape (batch, num_criteria) ground truth ratings
            overall_ratings  (Tensor): Shape (batch,) ground truth overall rating
            lambda_reg       (float) : L2 weight decay parameter

        Returns:
            total_loss (Tensor): Scalar training loss
        """
        device = user_ids.device
        batch_size = user_ids.shape[0]
        r_levels = torch.arange(1, self.num_ratings, device=device).float().unsqueeze(0)

        # 1. Independent Factorizations: criteria scores and vectors
        criterion_vectors, criterion_scores = self.forward_criteria(user_ids, item_ids)

        # 2. Criteria Smooth Hinge Losses
        crit_loss_sum = 0.0
        for c in range(self.num_criteria):
            thresh_c = self.get_coupled_thresholds(user_ids, target_idx=c)
            s_c = criterion_scores[:, c].unsqueeze(1).expand_as(thresh_c)
            y_c = criteria_ratings[:, c].float().unsqueeze(1).expand_as(thresh_c)

            T_c = torch.where(r_levels >= y_c, torch.ones_like(thresh_c), -torch.ones_like(thresh_c))
            z_c = T_c * (thresh_c - s_c)
            crit_loss_sum = crit_loss_sum + smooth_hinge_loss(z_c).sum()

        crit_loss_avg = crit_loss_sum / float(self.num_criteria)

        # 3. Ranking Aggregator forward pass
        overall_scores, attn_weights = self.ranking_aggregator(criterion_vectors, criterion_scores)

        # 4. Overall Rating Smooth Hinge Loss using coupled overall thresholds
        thresh_overall = self.get_coupled_thresholds(user_ids, target_idx=self.num_criteria)
        s_overall = overall_scores.unsqueeze(1).expand_as(thresh_overall)
        y_overall = overall_ratings.float().unsqueeze(1).expand_as(thresh_overall)

        T_overall = torch.where(r_levels >= y_overall, torch.ones_like(thresh_overall), -torch.ones_like(thresh_overall))
        z_overall = T_overall * (thresh_overall - s_overall)
        overall_hinge_loss = smooth_hinge_loss(z_overall).sum()

        # 5. BPR Ranking Loss on aggregated scores
        bpr_loss = self.compute_bpr_loss(user_ids, overall_scores, item_ids, overall_ratings)

        # 6. L2 Regularization (Frobenius norm over all parameters)
        l2_loss = 0.0
        for p in self.parameters():
            l2_loss = l2_loss + (p ** 2).sum()
        l2_loss = (lambda_reg / 2.0) * l2_loss

        # 7. Total Joint Loss
        total_loss = overall_hinge_loss + (self.criteria_alpha * crit_loss_avg) + (self.bpr_alpha * bpr_loss) + l2_loss

        return total_loss

    def predict_scores(self, user_ids, item_ids):
        """
        Predict continuous overall interaction scores for ranking (higher = more preferred).

        Parameters:
            user_ids (Tensor): Shape (N,)
            item_ids (Tensor): Shape (N,)

        Returns:
            Tensor: Shape (N,) continuous ranking scores S_hat_{ij}
        """
        with torch.no_grad():
            criterion_vectors, criterion_scores = self.forward_criteria(user_ids, item_ids)
            overall_scores, _ = self.ranking_aggregator(criterion_vectors, criterion_scores)
        return overall_scores

    def predict_ratings(self, user_ids, item_ids):
        """
        Predict ordinal overall ratings {1, ..., R} by comparing aggregated score S_hat_{ij}
        against coupled overall thresholds Theta_tilde_{i, r}^(overall).

        Parameters:
            user_ids (Tensor): Shape (N,)
            item_ids (Tensor): Shape (N,)

        Returns:
            Tensor: Shape (N,) predicted integer ratings in {1, ..., R}
        """
        with torch.no_grad():
            overall_scores = self.predict_scores(user_ids, item_ids)
            thresh_overall = self.get_coupled_thresholds(user_ids, target_idx=self.num_criteria)
            scores_expanded = overall_scores.unsqueeze(1).expand_as(thresh_overall)
            exceeds = (scores_expanded > thresh_overall).sum(dim=1)
            predicted = (exceeds + 1).clamp(1, self.num_ratings)
        return predicted

    def predict_criteria_ratings(self, user_ids, item_ids):
        """
        Predict ordinal ratings for each of the 7 criteria independently.

        Parameters:
            user_ids (Tensor): Shape (N,)
            item_ids (Tensor): Shape (N,)

        Returns:
            Tensor: Shape (N, num_criteria) predicted integer ratings
        """
        with torch.no_grad():
            _, criterion_scores = self.forward_criteria(user_ids, item_ids)
            all_preds = []
            for c in range(self.num_criteria):
                thresh_c = self.get_coupled_thresholds(user_ids, target_idx=c)
                s_c_exp = criterion_scores[:, c].unsqueeze(1).expand_as(thresh_c)
                exceeds = (s_c_exp > thresh_c).sum(dim=1)
                pred_c = (exceeds + 1).clamp(1, self.num_ratings)
                all_preds.append(pred_c)
        return torch.stack(all_preds, dim=1)

    def get_criterion_attention_weights(self, user_ids, item_ids):
        """
        Return dynamic attention weights assigned to each criterion by the ranking aggregator.

        Parameters:
            user_ids (Tensor): Shape (N,)
            item_ids (Tensor): Shape (N,)

        Returns:
            Tensor: Shape (N, num_criteria) attention distribution summing to 1.0 per sample
        """
        with torch.no_grad():
            criterion_vectors, criterion_scores = self.forward_criteria(user_ids, item_ids)
            _, attn_weights = self.ranking_aggregator(criterion_vectors, criterion_scores)
        return attn_weights

    def forward(self, user_ids, item_ids):
        """Inference forward pass returning predicted overall ratings as floats."""
        return self.predict_ratings(user_ids, item_ids).float()


