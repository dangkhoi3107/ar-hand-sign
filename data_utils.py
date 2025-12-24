# data_utils.py
import numpy as np

def normalize_landmarks(data):
    """
    Chuẩn hóa dữ liệu đầu vào để đồng nhất giữa Web và Video.
    Input: Array (N, 3) hoặc (N*3,)
    Output: Vector (63,) đã chuẩn hóa (Scale Invariant)
    """
    pts = np.array(data, dtype=np.float32).reshape(-1, 3)

    # 1. Dời gốc tọa độ về cổ tay (điểm số 0)
    wrist = pts[0].copy()
    pts = pts - wrist

    # 2. Scale: Chia cho khoảng cách từ cổ tay đến đốt ngón giữa (điểm 9)
    # Giống logic trong extract_sequences_from_videos.py của bạn
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

    return pts.flatten() # Trả về vector 63 chiều

def pad_sequence(seq, target_len=30):
    """Đảm bảo chuỗi luôn có độ dài cố định (cắt hoặc thêm số 0)"""
    seq = np.array(seq, dtype=np.float32)
    length = len(seq)
    feat_dim = seq.shape[1]

    if length == target_len:
        return seq
    if length > target_len:
        # Resample nếu dài hơn (lấy mẫu đều)
        idx = np.linspace(0, length - 1, target_len).astype(int)
        return seq[idx]

    # Padding số 0 nếu ngắn hơn
    padding = np.zeros((target_len - length, feat_dim), dtype=np.float32)
    return np.vstack([seq, padding])