import torch
import torch.nn.functional as F
import math

def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries: (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys: (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values: (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    # Транспонируем keys для матричного умножения с батчем
    keys_transposed = keys.transpose(-2, -1)  # Размер: (BATCH_SIZE, HIDDEN_DIM, SEQ_LENGTH)
    
    # Вычисляем исходные оценки внимания
    scores = torch.matmul(queries, keys_transposed)  # Размер: (BATCH_SIZE, SEQ_LENGTH, SEQ_LENGTH)
    
    # Масштабируем оценки на корень из скрытого размера
    d_k = queries.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # Применяем softmax для получения весов внимания
    attn_weights = F.softmax(scores, dim=-1)  # Размер: (BATCH_SIZE, SEQ_LENGTH, SEQ_LENGTH)
    
    # Вычисляем выход внимания
    output = torch.matmul(attn_weights, values)  # Размер: (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    
    return output

def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries: (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys: (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values: (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix: (N_HEADS * DIM_PER_HEAD, N_HEADS * DIM_PER_HEAD)
    """
    # Вычисляем attention с масштабированием скалярного произведения для каждой головы
    d_k = queries.size(-1)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)  # (BATCH_SIZE, N_HEADS, SEQ_LENGTH, SEQ_LENGTH)
    attn_weights = F.softmax(scores, dim=-1)  # (BATCH_SIZE, N_HEADS, SEQ_LENGTH, SEQ_LENGTH)
    multihead_output = torch.matmul(attn_weights, values)  # (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    
    # Преобразуем и применяем проекционную матрицу для объединения голов
    batch_size, n_heads, seq_length, dim_per_head = multihead_output.shape
    combined_output = multihead_output.transpose(1, 2).reshape(batch_size, seq_length, -1)  # (BATCH_SIZE, SEQ_LENGTH, N_HEADS * DIM_PER_HEAD)
    final_output = torch.matmul(combined_output, projection_matrix)  # (BATCH_SIZE, SEQ_LENGTH, N_HEADS * DIM_PER_HEAD)
    
    return final_output

def compute_rotary_embeddings(x) -> torch.Tensor:
    """
    x: (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    batch_size, seq_length, n_heads, dim_per_head = x.shape
    half_dim = dim_per_head // 2
    
    # Определяем частоты углов
    angle_rates = torch.pow(10000, -torch.arange(half_dim, dtype=torch.float32) / half_dim)
    angles = torch.einsum('i,j->ij', torch.arange(seq_length, dtype=torch.float32), angle_rates)
    
    # Вычисляем синус и косинус для embeddings
    sin_embedding = torch.sin(angles).unsqueeze(0).unsqueeze(2)  # (1, SEQ_LENGTH, 1, HALF_DIM)
    cos_embedding = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # (1, SEQ_LENGTH, 1, HALF_DIM)
    
    # Применяем rotary embeddings
    x_sin, x_cos = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([x_sin * cos_embedding - x_cos * sin_embedding,
                           x_sin * sin_embedding + x_cos * cos_embedding], dim=-1)
    
    return x_rotated
