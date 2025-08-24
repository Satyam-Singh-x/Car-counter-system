"""
SORT: A Simple, Online and Realtime Tracker
Modified version without FilterPy dependency
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict

class KalmanFilter:
    """A simple Kalman filter for tracking bounding boxes in image space."""
    
    def __init__(self):
        # State vector: [x, y, s, r, ẋ, ẏ, ṡ]
        # where x, y are center coordinates, s is scale/area, r is aspect ratio
        self.ndim = 4  # Measurement dimension
        self.dt = 1.0  # Time step
        
        # State transition matrix
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i+4] = self.dt
        
        # Measurement matrix
        self.H = np.eye(4, 7)
        
        # Initialize state covariance
        self.P = np.eye(7) * 10
        
        # Process noise covariance
        self.Q = np.eye(7) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(4) * 1
        
    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        x, y, w, h = measurement
        return np.array([x, y, w * h, w / h, 0, 0, 0]), self.P.copy()
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        # Predict state
        mean = np.dot(self.F, mean)
        # Predict covariance
        covariance = np.dot(np.dot(self.F, covariance), self.F.T) + self.Q
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        # Project mean
        mean = np.dot(self.H, mean)
        # Project covariance
        covariance = np.dot(np.dot(self.H, covariance), self.H.T) + self.R
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        # Project state to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Kalman gain
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.lstsq(
            chol_factor.T,
            np.linalg.lstsq(chol_factor, np.dot(covariance, self.H.T).T)[0].T,
            rcond=None
        )[0].T
        
        # Update state mean
        mean = mean + np.dot(kalman_gain, measurement - projected_mean)
        
        # Update state covariance
        covariance = covariance - np.dot(kalman_gain, np.dot(projected_cov, kalman_gain.T))
        
        return mean, covariance

def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two sets of bounding boxes.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    iou = intersection / (area_test + area_gt - intersection + 1e-8)
    return iou

class Track:
    """A single target track with state, id, and tracking history."""
    
    def __init__(self, mean, covariance, track_id, n_init, max_age):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.state = 'tentative' if n_init > 0 else 'confirmed'
        self._n_init = n_init
        self._max_age = max_age
        
    def predict(self, kf):
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf, detection):
        """Perform Kalman filter measurement update."""
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection)
        self.hits += 1
        self.time_since_update = 0
        
        if self.state == 'tentative' and self.hits >= self._n_init:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > self._max_age:
            self.state = 'deleted'
    
    def is_tentative(self):
        """Return True if this track is tentative (unconfirmed)."""
        return self.state == 'tentative'
    
    def is_confirmed(self):
        """Return True if this track is confirmed."""
        return self.state == 'confirmed'
    
    def is_deleted(self):
        """Return True if this track is dead and should be deleted."""
        return self.state == 'deleted'

class Sort:
    """
    SORT: A Simple, Online and Realtime Tracker
    """
    
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections):
        """
        Parameters
        ----------
        detections : ndarray
            An N x 5 array of detections in the format [[x1, y1, x2, y2, score], ...]
        
        Returns
        -------
        ndarray
            An M x 5 array of tracked objects in the format [[x1, y1, x2, y2, id], ...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing tracks
        trks = []
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.tracks):
            # Predict current track location with Kalman filter
            trk.predict(self.kf)
            
            # Convert state to bounding box
            pos = trk.mean
            w = np.sqrt(pos[2] * pos[3])
            h = pos[2] / w
            x1 = pos[0] - w / 2
            y1 = pos[1] - h / 2
            x2 = pos[0] + w / 2
            y2 = pos[1] + h / 2
            
            trks.append([x1, y1, x2, y2])
        
        # Convert to numpy array
        trks = np.array(trks) if len(trks) > 0 else np.empty((0, 4))
        
        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(
            detections[:, :4], trks
        )
        
        # Update matched tracks with assigned detections
        for m in matched:
            self.tracks[m[1]].update(self.kf, detections[m[0], :4])
        
        # Create and initialize new tracks for unmatched detections
        for i in unmatched_dets:
            mean, cov = self.kf.initiate(detections[i, :4])
            self.tracks.append(Track(mean, cov, self.next_id, self.min_hits, self.max_age))
            self.next_id += 1
        
        # Update unmatched tracks
        for i in unmatched_trks:
            self.tracks[i].mark_missed()
        
        # Return confirmed tracks
        for trk in self.tracks:
            if trk.is_confirmed() and trk.time_since_update == 0:
                # Convert state to bounding box
                pos = trk.mean
                w = np.sqrt(pos[2] * pos[3])
                h = pos[2] / w
                x1 = pos[0] - w / 2
                y1 = pos[1] - h / 2
                x2 = pos[0] + w / 2
                y2 = pos[1] + h / 2
                
                ret.append([x1, y1, x2, y2, trk.track_id])
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 5))
    
    def associate_detections_to_tracks(self, detections, tracks):
        """Assigns detections to tracked objects."""
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        # Compute IoU between detections and tracks
        iou_matrix = iou_batch(detections, tracks)
        
        # Solve assignment problem
        if min(iou_matrix.shape) > 0:
            # Use Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter out matches with low IoU
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t in range(len(tracks)):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)
        
        # Filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_tracks)
