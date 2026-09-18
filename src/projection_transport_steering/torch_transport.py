import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from projection_transport_steering.transport import EmpiricalQuantileTransport


def fit_fisher_basis(gradients: Tensor, rank: int) -> Tensor:
    if (
        gradients.ndim != 2
        or not torch.all(torch.isfinite(gradients))
        or not isinstance(rank, int)
        or rank < 1
        or rank > min(gradients.shape)
    ):
        raise ValueError("gradients and rank are invalid")
    _, _, right = torch.linalg.svd(gradients, full_matrices=False)
    return right[:rank].transpose(0, 1).contiguous()


def restricted_metric_steering(
    basis: Tensor,
    restricted_metric: Tensor,
    target_covector: Tensor,
    progress: float,
) -> tuple[Tensor, Tensor]:
    if (
        basis.ndim != 2
        or target_covector.ndim != 1
        or basis.shape[0] != target_covector.shape[0]
        or restricted_metric.shape != (basis.shape[1], basis.shape[1])
        or basis.shape[1] == 0
        or not isinstance(progress, (int, float))
        or isinstance(progress, bool)
        or not math.isfinite(progress)
        or not basis.is_floating_point()
        or not restricted_metric.is_floating_point()
        or not target_covector.is_floating_point()
        or basis.device != restricted_metric.device
        or basis.device != target_covector.device
        or not torch.all(torch.isfinite(basis))
        or not torch.all(torch.isfinite(restricted_metric))
        or not torch.all(torch.isfinite(target_covector))
    ):
        raise ValueError("restricted steering inputs are invalid")
    work_dtype = torch.float64 if torch.float64 in {
        basis.dtype,
        restricted_metric.dtype,
        target_covector.dtype,
    } else torch.float32
    work_basis = basis.to(work_dtype)
    work_metric = restricted_metric.to(work_dtype)
    work_covector = target_covector.to(work_dtype)
    if torch.linalg.matrix_rank(work_basis) != work_basis.shape[1] or not torch.allclose(
        work_metric,
        work_metric.transpose(0, 1),
    ):
        raise ValueError("basis or restricted metric is invalid")
    factor, info = torch.linalg.cholesky_ex(work_metric)
    if torch.any(info != 0):
        raise ValueError("restricted metric must be positive definite")
    coordinates = work_basis.transpose(0, 1) @ work_covector
    inverse_coordinates = torch.cholesky_solve(coordinates.unsqueeze(1), factor).squeeze(1)
    inverse_quadratic = torch.dot(coordinates, inverse_coordinates)
    if not torch.isfinite(inverse_quadratic) or inverse_quadratic <= 0:
        raise ValueError("target covector is not identified in the subspace")
    update = float(progress) * work_basis @ inverse_coordinates / inverse_quadratic
    return update.to(basis.dtype), inverse_quadratic


def _validate_fisher_metric(velocity: Tensor, basis: Tensor, penalty: float) -> None:
    if (
        velocity.ndim < 1
        or basis.ndim != 2
        or basis.shape[0] != velocity.shape[-1]
        or basis.shape[1] == 0
        or not math.isfinite(penalty)
        or penalty < 0.0
        or not torch.all(torch.isfinite(velocity))
        or not torch.all(torch.isfinite(basis))
    ):
        raise ValueError("velocity, basis, and penalty are invalid")
    work_basis = basis.float()
    gram = work_basis.transpose(0, 1) @ work_basis
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
        raise ValueError("protected basis must have orthonormal columns")


def fisher_metric_filtered_velocity(velocity: Tensor, basis: Tensor, penalty: float) -> Tensor:
    _validate_fisher_metric(velocity, basis, penalty)
    original_dtype = velocity.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    values = velocity.to(work_dtype)
    protected = basis.to(device=values.device, dtype=work_dtype)
    inverse_values = values - (penalty / (1.0 + penalty)) * (
        (values @ protected) @ protected.transpose(0, 1)
    )
    return inverse_values.to(original_dtype)


def fisher_protected_velocity(velocity: Tensor, basis: Tensor, penalty: float) -> Tensor:
    original_dtype = velocity.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    values = velocity.to(work_dtype)
    inverse_values = fisher_metric_filtered_velocity(velocity, basis, penalty).to(work_dtype)
    raw_progress = values.square().sum(dim=-1, keepdim=True)
    inverse_progress = (values * inverse_values).sum(dim=-1, keepdim=True)
    scale = torch.where(raw_progress == 0.0, 1.0, raw_progress / inverse_progress)
    return (scale * inverse_values).to(original_dtype)


class FisherProtectedFlow(nn.Module):
    def __init__(
        self,
        flow: nn.Module,
        basis: Tensor,
        penalty: float,
        preserve_progress: bool = True,
    ) -> None:
        super().__init__()
        _validate_fisher_metric(torch.zeros(basis.shape[0]), basis, penalty)
        self.flow = flow
        self.basis: Tensor
        self.register_buffer("basis", basis)
        self.penalty = penalty
        self.preserve_progress = preserve_progress

    def forward(self, *args: object, **kwargs: object) -> tuple[Tensor, object]:
        velocity, caches = self.flow(*args, **kwargs)
        transform = (
            fisher_protected_velocity
            if self.preserve_progress
            else fisher_metric_filtered_velocity
        )
        return transform(velocity, self.basis, self.penalty), caches


class TorchEmpiricalQuantileTransport(nn.Module):
    def __init__(
        self,
        source_knots: Tensor,
        target_knots: Tensor,
        left: float,
        right: float,
    ) -> None:
        super().__init__()
        self.register_buffer("source_knots", source_knots)
        self.register_buffer("target_knots", target_knots)
        self.left = left
        self.right = right

    @classmethod
    def from_empirical(
        cls,
        transport: EmpiricalQuantileTransport,
    ) -> "TorchEmpiricalQuantileTransport":
        return cls(
            torch.from_numpy(transport.source_knots),
            torch.from_numpy(transport.target_knots),
            transport.left,
            transport.right,
        )

    def forward(self, values: Tensor) -> Tensor:
        source = self.source_knots.to(device=values.device, dtype=values.dtype)
        target = self.target_knots.to(device=values.device, dtype=values.dtype)
        flat = values.reshape(-1)
        upper = torch.searchsorted(source, flat)
        lower = torch.clamp(upper - 1, 0, source.numel() - 1)
        upper = torch.clamp(upper, 0, source.numel() - 1)
        denominator = source[upper] - source[lower]
        weight = torch.where(denominator == 0, 0, (flat - source[lower]) / denominator)
        mapped = target[lower] + weight * (target[upper] - target[lower])
        mapped = torch.where(flat < source[0], flat.new_tensor(self.left), mapped)
        mapped = torch.where(flat > source[-1], flat.new_tensor(self.right), mapped)
        return mapped.reshape(values.shape)


class ProjectionTransportAction(nn.Module):
    def __init__(
        self,
        basis: Tensor,
        inverse_metric_basis: Tensor,
        gram_inverse: Tensor,
        transport: TorchEmpiricalQuantileTransport,
        strength: float,
    ) -> None:
        super().__init__()
        self.register_buffer("basis", basis)
        self.register_buffer("inverse_metric_basis", inverse_metric_basis)
        self.register_buffer("gram_inverse", gram_inverse)
        self.transport = transport
        self.strength = strength

    @classmethod
    def from_empirical(
        cls,
        basis: Tensor,
        metric: Tensor,
        transport: EmpiricalQuantileTransport,
        strength: float,
    ) -> "ProjectionTransportAction":
        if basis.ndim != 2 or basis.shape[1] != 1:
            raise ValueError("empirical projection transport requires a rank-one basis")
        if metric.shape != (basis.shape[0], basis.shape[0]):
            raise ValueError("metric shape is incompatible with basis")
        torch.linalg.cholesky(metric)
        inverse_metric_basis = torch.linalg.solve(metric, basis)
        gram = basis.transpose(0, 1) @ inverse_metric_basis
        if torch.linalg.matrix_rank(gram) != 1:
            raise ValueError("basis is not identified under the metric")
        return cls(
            basis,
            inverse_metric_basis,
            torch.linalg.inv(gram),
            TorchEmpiricalQuantileTransport.from_empirical(transport),
            strength,
        )

    @classmethod
    def euclidean(
        cls,
        basis: Tensor,
        transport: EmpiricalQuantileTransport,
        strength: float,
    ) -> "ProjectionTransportAction":
        if basis.ndim != 2 or basis.shape[1] != 1:
            raise ValueError("empirical projection transport requires a rank-one basis")
        gram = basis.transpose(0, 1) @ basis
        if torch.linalg.matrix_rank(gram) != 1:
            raise ValueError("basis is not identified")
        return cls(
            basis,
            basis,
            torch.linalg.inv(gram),
            TorchEmpiricalQuantileTransport.from_empirical(transport),
            strength,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        basis = self.basis.to(device=hidden.device, dtype=hidden.dtype)
        inverse_metric_basis = self.inverse_metric_basis.to(device=hidden.device, dtype=hidden.dtype)
        gram_inverse = self.gram_inverse.to(device=hidden.device, dtype=hidden.dtype)
        projections = hidden @ basis
        mapped = self.transport(projections)
        targets = projections + self.strength * (mapped - projections)
        coefficients = (targets - projections) @ gram_inverse
        return hidden + coefficients @ inverse_metric_basis.transpose(0, 1)


class ScoreConditionedAction(nn.Module):
    def __init__(
        self,
        action: nn.Module,
        direction: Tensor,
        slope: float,
        intercept: float,
        threshold: float,
    ) -> None:
        super().__init__()
        if direction.ndim != 1 or not 0.0 <= threshold <= 1.0:
            raise ValueError("direction and threshold are invalid")
        self.action = action
        self.register_buffer("direction", direction)
        self.slope = slope
        self.intercept = intercept
        self.threshold = threshold

    def forward(self, hidden: Tensor) -> Tensor:
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        score = torch.sigmoid(self.slope * (hidden @ direction) + self.intercept)
        selected = score >= self.threshold
        return torch.where(selected.unsqueeze(-1), self.action(hidden), hidden)


class GaussianCoordinateTransportAction(nn.Module):
    def __init__(
        self,
        source_mean: Tensor,
        target_mean: Tensor,
        scale: Tensor,
        mask: Tensor,
        strength: float,
    ) -> None:
        super().__init__()
        self.register_buffer("source_mean", source_mean)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("scale", scale)
        self.register_buffer("mask", mask)
        self.strength = strength

    @classmethod
    def fit(
        cls,
        source: Tensor,
        target: Tensor,
        strength: float,
        epsilon: float = 1e-4,
    ) -> "GaussianCoordinateTransportAction":
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("source and target must be feature-aligned matrices")
        source_mean = source.mean(0)
        target_mean = target.mean(0)
        source_std = source.std(0)
        target_std = target.std(0)
        mask = (source_std > epsilon) & (target_std > epsilon)
        scale = torch.ones_like(source_std)
        scale[mask] = target_std[mask] / source_std[mask]
        return cls(source_mean, target_mean, scale, mask, strength)

    def forward(self, hidden: Tensor) -> Tensor:
        source_mean = self.source_mean.to(device=hidden.device, dtype=hidden.dtype)
        target_mean = self.target_mean.to(device=hidden.device, dtype=hidden.dtype)
        scale = self.scale.to(device=hidden.device, dtype=hidden.dtype)
        mask = self.mask.to(device=hidden.device)
        transported = target_mean + scale * (hidden - source_mean)
        moved = hidden + self.strength * (transported - hidden)
        return torch.where(mask, moved, hidden)


class FullCovarianceTransportAction(nn.Module):
    def __init__(
        self,
        source_mean: Tensor,
        target_mean: Tensor,
        matrix: Tensor,
        strength: float,
    ) -> None:
        super().__init__()
        self.register_buffer("source_mean", source_mean)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("matrix", matrix)
        self.strength = strength

    @staticmethod
    def _power(matrix: Tensor, exponent: float) -> Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        if torch.any(eigenvalues <= 0):
            raise ValueError("regularized covariance must be positive definite")
        return (eigenvectors * eigenvalues.pow(exponent)) @ eigenvectors.transpose(0, 1)

    @classmethod
    def fit(
        cls,
        source: Tensor,
        target: Tensor,
        strength: float,
        regularization: float,
    ) -> "FullCovarianceTransportAction":
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape[1] != target.shape[1]
            or min(source.shape[0], target.shape[0]) < 2
            or regularization <= 0
        ):
            raise ValueError("samples and regularization are invalid")
        source_values = source.double()
        target_values = target.double()
        source_mean = source_values.mean(0)
        target_mean = target_values.mean(0)
        identity = torch.eye(source.shape[1], device=source.device, dtype=torch.float64)
        source_covariance = torch.cov(source_values.T) + regularization * identity
        target_covariance = torch.cov(target_values.T) + regularization * identity
        source_root = cls._power(source_covariance, 0.5)
        source_inverse_root = cls._power(source_covariance, -0.5)
        middle_root = cls._power(source_root @ target_covariance @ source_root, 0.5)
        matrix = source_inverse_root @ middle_root @ source_inverse_root
        return cls(source_mean, target_mean, matrix, strength)

    def forward(self, hidden: Tensor) -> Tensor:
        source_mean = self.source_mean.to(device=hidden.device, dtype=hidden.dtype)
        target_mean = self.target_mean.to(device=hidden.device, dtype=hidden.dtype)
        matrix = self.matrix.to(device=hidden.device, dtype=hidden.dtype)
        transported = (hidden - source_mean) @ matrix.transpose(0, 1) + target_mean
        return hidden + self.strength * (transported - hidden)


class SphericalSteeringAction(nn.Module):
    def __init__(
        self,
        target: Tensor,
        source: Tensor,
        kappa: float,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()
        if target.ndim != 1 or source.shape != target.shape or not 0.0 <= alpha <= 1.0:
            raise ValueError("spherical prototypes or strength are invalid")
        self.register_buffer("target", F.normalize(target.float(), dim=0))
        self.register_buffer("source", F.normalize(source.float(), dim=0))
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def forward(self, hidden: Tensor) -> Tensor:
        original_dtype = hidden.dtype
        values = hidden.float()
        target = self.target.to(values.device)
        source = self.source.to(values.device)
        norms = values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = values / norms
        target_cosine = (unit * target).sum(-1).clamp(-1.0, 1.0)
        source_cosine = (unit * source).sum(-1).clamp(-1.0, 1.0)
        probabilities = torch.softmax(
            self.kappa * torch.stack((target_cosine, source_cosine), dim=-1),
            dim=-1,
        )
        delta = probabilities[..., 1] - probabilities[..., 0]
        amount = (self.alpha * (delta - self.beta) / (1.0 - self.beta)).clamp(0.0, 1.0)
        theta = torch.acos(target_cosine)
        sine = torch.sin(theta)
        safe_sine = sine.clamp_min(1e-7)
        orthogonal = (unit - target_cosine.unsqueeze(-1) * target) / safe_sine.unsqueeze(-1)
        new_theta = (1.0 - amount) * theta
        rotated = (
            torch.cos(new_theta).unsqueeze(-1) * target
            + torch.sin(new_theta).unsqueeze(-1) * orthogonal
        ) * norms
        selected = (delta > self.beta) & (theta >= 1e-4)
        return torch.where(selected.unsqueeze(-1), rotated, values).to(original_dtype)
