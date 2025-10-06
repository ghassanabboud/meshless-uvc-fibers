from Sparse import *
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_local_system(apicobasal_vectors_sparse, transmural_vectors_sparse):
    """
    Calculate the local coordinate system based on the gradients of apicobasal coordinate and the transmural coordinate (rho_f in Biasi et al.)
    
    Parameters
    ----------
    apicobasal_vectors_sparse : SparseVectors
        SparseVectors containing the apicobasal coordinate vectors.
    transmural_vectors_sparse : SparseVectors
        SparseVectors containing the transmural coordinate vectors.

    Returns
    -------
    tuple
        e_l, e_t, e_c : (SparseVectors, SparseVectors, SparseVectors)
            The local coordinate system vectors.
    """
    if apicobasal_vectors_sparse.device != transmural_vectors_sparse.device:
        raise ValueError("Sparse_Apicobasal and Sparse_Transmural must be on the same device.")
    
    apico_vectors = apicobasal_vectors_sparse.vectors.flip(1)
    transmural_vectors = transmural_vectors_sparse.vectors.flip(1)

    dot_product = torch.sum(apico_vectors * transmural_vectors, axis=-1)
    to_subtract = dot_product[..., None] * apico_vectors
    numerator = transmural_vectors - to_subtract
    norms = torch.linalg.norm(numerator, axis=-1)
    norms[norms == 0] = 1  # Avoid division by zero
    e_t = numerator / norms[..., None]
    e_c = torch.linalg.cross(apico_vectors, e_t)

    e_t = e_t.flip(1)
    e_c = e_c.flip(1)

    e_t = SparseVectors.from_Sparse_with_vectors(transmural_vectors_sparse, e_t)
    e_c = SparseVectors.from_Sparse_with_vectors(transmural_vectors_sparse, e_c)

    return apicobasal_vectors_sparse, e_t, e_c

def get_alpha_values(transmural, alpha_endo=60, alpha_epi=-60):
    """
    Calculate alpha values for rotating the local coordinate system based on transmural and transventricular coords.
    
    Parameters
    ----------
    transmural : SparseField
        SparseField containing the simple transmural coordinates
    alpha_endo : float, optional
        Alpha value for the LV endocardium, by default 60.
    alpha__epi : float, optional
        Alpha value for the LV epicardium, by default -60.

    Returns
    -------
    SparseField
        SparseField containing the alpha values for rotating the local coordinate system.
    """
    
    transmural_values = transmural.values
    alpha = (1-transmural_values) * alpha_endo + transmural_values * alpha_epi
    alpha = SparseField.from_Sparse_with_values(transmural, alpha)
    return alpha

def get_alpha_values_doste(transmural, w_function, alpha_endo=90, alpha_epi=-25, alpha_OT_endo=90, alpha_OT_epi=0):
    """
    Calculate alpha values for rotating the local coordinate system based on transmural and transventricular coords.
    
    Parameters
    ----------
    transmural : SparseField
        SparseField containing the simple transmural coordinates
    w_function : SparseField
        SparseField containing the w function values for the Doste methodology. 0 in the pulmonary valve and 1 in the tricuspid valve and the apex.
    alpha_endo : float, optional
        Alpha value for the RV endocardium, by default 90.
    alpha_epi : float, optional
        Alpha value for the RV epicardium, by default -25.
    alpha_OT_endo : float, optional
        Alpha value for the RVOT endocardium, by default 90.
    alpha_OT_epi : float, optional
        Alpha value for the RVOT epicardium, by default 0.

    Returns
    -------
    SparseField
        SparseField containing the alpha values for rotating the local coordinate system.
    """
    
    transmural_values = transmural.values
    w_values = w_function.values

    alpha_endo = (1 - w_values) * alpha_OT_endo + w_values * alpha_endo
    alpha_epi = (1 - w_values) * alpha_OT_epi + w_values * alpha_epi

    alpha = (1-transmural_values) * alpha_endo + transmural_values * alpha_epi
    alpha = SparseField.from_Sparse_with_values(transmural, alpha)
    return alpha


def get_beta_values(transmural, beta_endo=-20, beta_epi=20):
    """
    Calculate beta values for rotating the local coordinate system based on transmural and transventricular coords.
    
    Parameters
    ----------
    transmural_coords : SparseField
        SparseField containing the simple transmural coordinates
    beta_endo : float, optional
        Beta value for the LV endocardium, by default -20.
    beta_epi : float, optional
        Beta value for the LV epicardium, by default 20.

    Returns
    -------
    SparseField
        SparseField containing the beta values for rotating the local coordinate system.
    """
    
    transmural_values = transmural.values
    beta= (1 - transmural_values) * beta_endo + transmural_values * beta_epi
    beta = SparseField.from_Sparse_with_values(transmural, beta)
    return beta

def get_beta_values_doste(transmural, w_function, beta_endo=60, beta_epi=-60, beta_OT_endo=90, beta_OT_epi=0):
    """
    Calculate beta values for rotating the local coordinate system based on transmural and transventricular coords.
    
    Parameters
    ----------
    transmural : SparseField
        SparseField containing the simple transmural coordinates
    beta_endo : float, optional
        Alpha value for the LV endocardium, by default 60.
    beta__epi : float, optional
        Alpha value for the LV epicardium, by default -60.

    Returns
    -------
    SparseField
        SparseField containing the beta values for rotating the local coordinate system.
    """
    
    transmural_values = transmural.values
    w_values = w_function.values

    beta_endo = (1 - w_values) * beta_endo + w_values * beta_OT_endo
    beta_epi = (1 - w_values) * beta_epi + w_values * beta_OT_epi

    beta = (1-transmural_values) * beta_endo + transmural_values * beta_epi
    beta = SparseField.from_Sparse_with_values(transmural, beta)
    return beta


def calculate_fibers(e_c,e_l,e_t,alphas,betas, calculate_sheet=True, calculate_transverse=True):
    """
    Calculate the fiber, sheet and transverse vectors based on the local coordinate system and the alpha and beta values.
    
    Parameters
    ----------
    e_c : SparseVectors
        SparseVectors containing the e_c vectors from the local coordinate system.
    e_l : SparseVectors
        SparseVectors containing the e_l vectors from the local coordinate system.
    e_t : SparseVectors
        SparseVectors containing the e_t vectors from the local coordinate system.
    alphas : SparseField
        SparseField containing the alpha values for rotating the local coordinate system.
    betas : SparseField
        SparseField containing the beta values for rotating the local coordinate system.
    calculate_sheet : bool, optional
        whether to calculate the sheet normal or not. Default is True.
    calculate_transverse : bool, optional
        whether to calculate the transverse direction or not. Default is True.

    Returns
    -------
    tuple
        fiber, sheet, transverse : SparseVectors
            fiber direction, sheet normal, transverse direction as SparseVectors.
    """

    fiber = e_c.rotate(e_t, alphas)  # Fiber direction is e_c rotated by alphas around e_l
    sheet = e_l.rotate(fiber, betas) if calculate_sheet else None
    transverse = e_t.rotate(fiber, betas) if calculate_transverse else None

    return fiber,sheet,transverse