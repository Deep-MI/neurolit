#!/usr/bin/env python3


# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import optparse
import warnings

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import numpy.typing as npt
from lapy import TriaMesh
from scipy import sparse
from scipy.ndimage import binary_dilation

from neurolit._version import get_version_with_hash
from neurolit.utils.log import get_logger

logger = get_logger(__name__)


HELPTEXT = """
Script to sample labels from image to surface and clean up. 

USAGE:
lesion_to_surface --inseg <segimg> --insurf <surf> --incort <cortex.label>
            --seglut <seglut> --surflut <surflut> --out_annot <out_annot>
            --projmm <float> --radius <float> --to_annot <annot>
            --dilation <int> --report <report_path>

Author: Clemens Pollak
Date: October 29, 2025
Based on FastSurfer's sample_parc.py and smooth_aparc.py by Martin Reuter (Dec-18-2023)

If you use neuroLIT for research publications, please cite:

Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole
  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience.

"""

h_inseg = "path to input segmentation image"
h_incort = "path to input cortex label mask"
h_insurf = "path to input surface"
h_out_annot = "path to output annot"
h_surflut = "FreeSurfer look-up-table for values on surface"
h_seglut = "Look-up-table for values in segmentation image (rows need to correspond to surflut). " \
                                "Label ID is the value of the label in the binarysegmentation image."
h_projmm = "Sample along normal at projmm distance (in mm), default 0"
h_radius = "Search around sample location at radius (in mm) for label if 'unknown', default None"
h_to_annot = "Replace annotations in existing annot file."
h_dilation = "Dilation radius for the lesion segmentation mask, default 3"
h_report = "Path to write the surface anatomy report (optional)"

def options_parse():
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options : argparse.Namespace
        Namespace object holding options.
    """
    parser = optparse.OptionParser(usage=HELPTEXT)
    parser.add_option("--inseg", dest="inseg", help=h_inseg)
    parser.add_option("--insurf", dest="insurf", help=h_insurf)
    parser.add_option("--incort", dest="incort", help=h_incort)
    parser.add_option("--surflut", dest="surflut", help=h_surflut)
    parser.add_option("--seglut", dest="seglut", help=h_seglut)
    parser.add_option("--out_annot", dest="out_annot", help=h_out_annot)
    parser.add_option("--projmm", dest="projmm", help=h_projmm, default=0.0, type="float")
    parser.add_option("--radius", dest="radius", help=h_radius, default=None, type="float")
    parser.add_option("--to_annot", dest="to_annot", help=h_to_annot)
    parser.add_option("--out_ctab", dest="out_ctab", help="path to output color table (ctab) file")
    parser.add_option("--dilation", dest="dilation", help=h_dilation, default=3, type="int")
    parser.add_option("--report", dest="report", help=h_report)
    parser.add_option("-v", "--version", dest="version", action="store_true", help="Print version number and exit")
    (options, args) = parser.parse_args()
    return options


def get_surface_anatomy_info(
    existing_labels: npt.NDArray[np.integer],
    target_mask: npt.NDArray[bool],
    adjM: sparse.csr_matrix
) -> tuple[set[int], set[int], set[int]]:
    """
    Categorize surface labels based on their relationship with the lesion mask.

    Parameters
    ----------
    existing_labels : npt.NDArray[np.integer]
        Original annotation labels (indices into ctab).
    target_mask : npt.NDArray[bool]
        Binary mask of the lesion on the surface.
    adjM : sparse.csr_matrix
        Adjacency matrix of the surface mesh.

    Returns
    -------
    replaced : Set[int]
        Label indices fully replaced by the lesion.
    reduced : Set[int]
        Label indices partially replaced by the lesion.
    adjacent : Set[int]
        Label indices touching the lesion boundary but not overlapping.
    """
    # Adjacency zone: neighbors of target_mask that are not in target_mask
    [II, JJ, VV] = sparse.find(adjM[target_mask, :])
    adjacent_indices = np.unique(JJ)
    adjacent_mask = np.zeros(existing_labels.shape, dtype=bool)
    adjacent_mask[adjacent_indices] = True
    adjacent_zone = adjacent_mask & ~target_mask

    labels_in_mask = set(np.unique(existing_labels[target_mask]))
    labels_outside_mask = set(np.unique(existing_labels[~target_mask]))
    labels_in_adj_zone = set(np.unique(existing_labels[adjacent_zone]))

    # Remove background/unknown (usually -1 or 0 in FS annotations)
    # FS labels from read_annot can be -1 for unknown
    for s in [labels_in_mask, labels_outside_mask, labels_in_adj_zone]:
        s.discard(-1)
        s.discard(0)

    # Categorize
    replaced = labels_in_mask - labels_outside_mask
    reduced = labels_in_mask & labels_outside_mask
    adjacent = labels_in_adj_zone - labels_in_mask

    return replaced, reduced, adjacent


def write_surface_report(
    output_path: str,
    replaced: set[int],
    reduced: set[int],
    adjacent: set[int],
    names: list
):
    """
    Write surface anatomy report to text file.

    Parameters
    ----------
    output_path : str
        Path to output text file.
    replaced : Set[int]
        Set of replaced label indices.
    reduced : Set[int]
        Set of reduced label indices.
    adjacent : Set[int]
        Set of adjacent label indices.
    names : list
        List of label names from the annotation.
    """
    # names might be bytes, decode if necessary
    decoded_names = []
    for n in names:
        if isinstance(n, bytes):
            decoded_names.append(n.decode('utf-8'))
        else:
            decoded_names.append(str(n))

    with open(output_path, 'w') as f:
        f.write("# Lesion Surface Anatomy Report\n")
        f.write("#" + "=" * 60 + "\n")
        f.write(f"# Number of replaced labels: {len(replaced)}\n")
        f.write(f"# Number of reduced labels:  {len(reduced)}\n")
        f.write(f"# Number of adjacent labels: {len(adjacent)}\n")
        f.write("#" + "=" * 60 + "\n\n")

        categories = [
            ("Replaced Labels (fully covered by lesion)", replaced),
            ("Reduced Labels (partially covered by lesion)", reduced),
            ("Adjacent Labels (touching lesion boundary)", adjacent)
        ]

        for title, labels in categories:
            f.write(f"# {title}\n")
            f.write("#" + "-" * 60 + "\n")
            if not labels:
                f.write("# None found\n")
            else:
                f.write("# Index    LabelName\n")
                for idx in sorted(labels):
                    name = decoded_names[int(idx)] if int(idx) < len(decoded_names) else "Unknown"
                    f.write(f"{int(idx):8d}    {name}\n")
            f.write("\n")

    logger.info(f"Surface anatomy report written to: {output_path}")


def mode_filter(
        adjM: sparse.csr_matrix,
        labels: npt.NDArray[np.integer],
        fillonlylabel = None,
        novote: npt.ArrayLike | None = None
) -> npt.NDArray[np.integer]:
    """
    Apply mode filter (smoothing) to integer labels on mesh vertices.

    Parameters
    ----------
    adjM : sparse.csr_matrix[bool]
        Symmetric adjacency matrix defining edges between vertices,
        this determines what edges can vote so usually one adds the
        identity to the adjacency matrix so that each vertex is included
        in its own vote.
    labels : npt.NDArray[np.integer]
        List of integer labels at each vertex of the mesh.
    fillonlylabel : int
        Label to fill exclusively. Defaults to None to smooth all labels.
    novote : npt.ArrayLike
        Label ids that should not vote. Defaults to None ([]).

    Returns
    -------
    labels_new : npt.NDArray[np.integer]
        New smoothed labels.
    """
    if novote is None:
        novote = []
    # make sure labels lengths equals adjM dimension
    n = labels.shape[0]
    if n != adjM.shape[0] or n != adjM.shape[1]:
        raise ValueError(
            "ERROR mode_filter: adjM size "
            + format(adjM.shape)
            + " does not match label length "
            + format(labels.shape)
        )
    # remove rows with only a single entry from adjM
    # if we removed some triangles, we may have isolated vertices
    # adding the eye to adjM will produce these entries
    # since they are neighbors to themselves, this adds
    # values to nlabels below that we don't want
    counts = np.diff(adjM.indptr)
    rows = np.where(counts == 1)
    pos = adjM.indptr[rows]
    adjM.data[pos] = 0
    adjM.eliminate_zeros()
    # for num rings exponentiate adjM and add adjM from step before
    # we currently do this outside of mode_filter
    # new labels will be the same as old almost everywhere
    labels_new = labels
    # find vertices to fill
    # if fillonlylabels empty, fill all
    if not fillonlylabel:
        ids = np.arange(0, n)
    else:
        # select the ones with the labels
        ids = np.where(labels == fillonlylabel)[0]
        if ids.size == 0:
            logger.warning(
                "No ids found with idx "
                + str(fillonlylabel)
                + "  ... continue"
            )
            return labels
    # of all ids to fill, find neighbors
    nbrs = adjM[ids, :]
    # get vertex ids (I, J ) of each edge in nbrs
    [II, JJ, VV] = sparse.find(nbrs)
    # check if we have neighbors with -1 or 0
    # this could produce problems in the loop below, so lets stop for now:
    nlabels = labels[JJ]
    # if any(nlabels == -1) or any(nlabels == 0):
    #     print('WARNING: neighbors have -1 or 0 labels!')
    #     #sys.exit("there are -1 or 0 labels in neighbors!")
    # create sparse matrix with labels at neighbors
    nlabels = sparse.csr_matrix((labels[JJ], (II, JJ)))
    # print("nlabels: {}".format(nlabels))
    from scipy.stats import mode

    if not isinstance(nlabels, sparse.csr_matrix):
        raise ValueError("Matrix must be CSR format.")
    # novote = [-1,0,fillonlylabel]
    # get rid of rows that have uniform vote (or are empty)
    # for this to work no negative numbers should exist
    # get row counts, max and sums
    # rmax = nlabels.max(1).A.squeeze()
    # sums = nlabels.sum(axis=1).A1
    # Convert sparse matrix operations to dense arrays for compatibility
    rmax = np.array(nlabels.max(1).todense()).flatten()
    sums = np.array(nlabels.sum(axis=1)).flatten()
    counts = np.diff(nlabels.indptr)
    # then keep rows where max*counts differs from sums
    rmax = np.multiply(rmax, counts)
    rows = np.where(rmax != sums)[0]
    # print("rows: " + str(nlabels.shape[0]) + "  reduced to " + str(rows.size))
    # Only after fixing the rows above, we can
    # get rid of entries that should not vote
    # since we have only rows that were non-uniform, they should not become empty
    # rows may become uniform: we still need to vote below to update this label
    if novote:
        rr = np.in1d(nlabels.data, novote)
        nlabels.data[rr] = 0
        nlabels.eliminate_zeros()
    # run over all rows and compute mode (maybe vectorize later)
    rempty = 0
    for row in rows:
        rvals = nlabels.data[nlabels.indptr[row] : nlabels.indptr[row + 1]]
        if rvals.size == 0:
            rempty += 1
            continue
        # print(str(rvals))
        mvals = mode(rvals, keepdims=True)[0]
        # print(str(mvals))
        if mvals.size != 0:
            # print(str(row)+' '+str(ids[row])+' '+str(mvals[0]))
            labels_new[ids[row]] = mvals[0]
    if rempty > 0:
        # should not happen
        logger.warning("row empty: " + str(rempty))
    # nbrs=np.squeeze(np.asarray(nbrs.todense())) # sparse matrix to dense matrix to np.array
    # nlabels=labels[nbrs]
    # counts = np.bincount(nlabels)
    # vote=np.argmax(counts)
    return labels_new


def get_adjM(trias: npt.NDArray[np.integer], n: int):
    """
    Create symmetric sparse adjacency matrix of triangle mesh.

    Parameters
    ----------
    trias : npt.NDArray[np.integer](m, 3)
        Triangle mesh matrix.
        
    n : int
        Shape of output (n,n) adjaceny matrix, where n>=m.

    Returns
    -------
    adjM : np.ndarray (bool) shape (n,n)
        Symmetric sparse CSR adjacency matrix, true corresponds to an edge.
    """
    T = trias
    J = T[:, [1, 2, 0]]
    # flatten
    T = T.flatten()
    J = J.flatten()
    adj = sparse.csr_matrix((np.ones(T.shape, dtype=bool), (T, J)), shape=(n, n))
    # if max adj is > 1 we have non manifold or mesh trias are not oriented
    # if matrix is not symmetric, we have a boundary
    # in case we have boundary, make sure this is a symmetric matrix
    adjM = (adj + adj.transpose()).astype(bool)
    return adjM


def sample_nearest_nonzero(img, vox_coords, radius=3.0):
    """
    Sample closest non-zero value in a ball of radius around vox_coords.

    Parameters
    ----------
    img : nibabel.image
        Image to sample. Voxels need to be isotropic.
    vox_coords : ndarray float shape(n,3)
        Coordinates in voxel space around which to search.
    radius : float default 3.0
        Consider all voxels inside this radius to find a non-zero value.

    Returns
    -------
    samples : np.ndarray(n,)
        Sampled values, returns zero for vertices where values are zero in ball.
    """
    # check for isotropic voxels 
    voxsize = img.header.get_zooms()
    logger.info(f"Check isotropic vox sizes: {voxsize}")
    assert (np.max(np.abs(voxsize - voxsize[0])) < 0.001), 'Voxels not isotropic!'
    data = np.asarray(img.dataobj)
    
    # radius in voxels:
    rvox = radius / voxsize[0]
    
    # sample window around nearest voxel
    x_nn = np.rint(vox_coords).astype(int)
    # Reason: to always have the same number of voxels that we check
    # and to be consistent with FreeSurfer, we center the window at
    # the nearest neighbor voxel, instead of at the float vox coordinates

    # create box with 2*rvox+1 side length to fully contain ball
    # and get coordinate offsets with zero at center
    ri = np.floor(rvox).astype(int)
    ll = np.arange(-ri,ri+1)
    xv, yv, zv = np.meshgrid(ll, ll, ll)
    # modify distances slightly, to avoid randomness when
    # sorting with different radius values for voxels that otherwise
    # have the same distance to center
    xvd = xv+0.001
    yvd = yv+0.002
    zvd = zv+0.003
    ddm = np.sqrt(xvd*xvd + yvd*yvd + zvd*zvd).flatten()
    # also compute correct distance for ball mask below
    dd = np.sqrt(xv*xv + yv*yv + zv*zv).flatten()
    ddball = dd<=rvox

    # flatten and keep only ball with radius
    xv = xv.flatten()[ddball]
    yv = yv.flatten()[ddball]
    zv = zv.flatten()[ddball]
    ddm = ddm[ddball]

    # stack to get offset vectors
    offsets = np.column_stack((xv, yv, zv))

    # sort offsets according to distance
    # Note: we keep the first zero voxel so we can later
    # determine if all voxels are zero with the argmax trick
    sortidx = np.argsort(ddm)
    offsets = offsets[sortidx,:]

    # reshape and tile to add to list of coords
    n = x_nn.shape[0]
    toffsets = np.tile(offsets.transpose().reshape(1,3,offsets.shape[0]),(n,1,1))
    s_coords = x_nn[:, :, np.newaxis] + toffsets

    # get image data at the s_coords locations
    s_data = data[s_coords[:,0], s_coords[:,1], s_coords[:,2]]

    # get first non-zero if possible
    nzidx = (s_data!=0).argmax(axis=1)
    # the above return index zero if all elements are zero which is OK for us
    # as we can then sample there and get a value of zero
    samples = s_data[np.arange(s_data.shape[0]),nzidx]
    return samples


def sample_img(surf, img, cortex=None, projmm=0.0, radius=None):
    """
    Sample volume at a distance from the surface.

    Parameters
    ----------
    surf : tuple | str
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.array of (n, 3) vertex coordinates and
        surf[1] is the np.array of (m, 3) triangle indices.
        If type is str, read surface from file.
    img : nibabel.image | str
        Image to sample.
        If type is str, read image from file.
    cortex : np.ndarray | str
        Filename of cortex label or np.array with cortex indices.
    projmm : float
        Sample projmm mm along the surface vertex normals (default=0).
    radius : float, optional 
        If given and if the sample is equal to zero, then consider
        all voxels inside this radius to find a non-zero value.

    Returns
    -------
    samples : np.ndarray (n,)
        Sampled values.
    """
    if isinstance(surf, str):
        surf = fs.read_geometry(surf, read_metadata=True)
    if isinstance(img, str):
        img = nib.load(img)
    if isinstance(cortex, str):
        cortex = fs.read_label(cortex)
    nvert = surf[0].shape[0]
    # Compute Cortex Mask
    if cortex is not None:
        mask = np.zeros(nvert, dtype=bool)
        mask[cortex] = True
    else:
        mask = np.ones(nvert, dtype=bool)

    data = np.asarray(img.dataobj)
    # Compute LaPy TriaMesh for vertex normal computation
    T = TriaMesh(surf[0], surf[1])
    # compute sample coordinates projmm mm along the surface normal
    # in surface RAS coordinate system:
    if not T.is_oriented():
        logger.warning("Surface is not oriented, orienting ...")
        T.orient_()
    x = T.v + projmm * T.vertex_normals()
    

    # mask cortex
    xx = x[mask]

    # compute Transformation from surface RAS to voxel space:
    Torig = img.header.get_vox2ras_tkr()
    Tinv = np.linalg.inv(Torig)
    x_vox = np.dot(xx, Tinv[:3, :3].T) + Tinv[:3, 3]
    # sample at nearest voxel
    x_nn = np.rint(x_vox).astype(int)
    samples_nn = data[x_nn[:,0], x_nn[:,1], x_nn[:,2]]
    # no search for zeros, done:
    if not radius:
        samplesfull = np.zeros(nvert, dtype="int")
        samplesfull[mask] = samples_nn
        return samplesfull
    # search for zeros, but no zeros exist, done:
    zeros = np.asarray(samples_nn==0).nonzero()[0]
    if zeros.size == 0:
        samplesfull = np.zeros(nvert, dtype="int")
        samplesfull[mask] = samples_nn
        return samplesfull
    # here we need to do the hard work of searching in a windows
    # for non-zero samples
    logger.info(f"sample_img: found {zeros.size} zero samples, searching radius ...")
    z_nn = x_nn[zeros]
    z_samples = sample_nearest_nonzero(img, z_nn, radius=radius)
    samples_nn[zeros] = z_samples
    samplesfull = np.zeros(nvert, dtype="int")
    samplesfull[mask] = samples_nn
    return samplesfull


def replace_labels(img_labels, img_lut, surf_lut):
    """
    Replace image labels with corresponding surface labels or unknown.

    Parameters
    ----------
    img_labels : np.ndarray(n,)
        Array with image label ids.
    img_lut : str
        Filename for image label look up table.
    surf_lut : str
        Filename for surface label look up table.

    Returns
    -------
    surf_labels : np.ndarray (n,)
        Array with surface label ids.
    surf_ctab : np.ndarray shape(m,4)
        Surface color table (RGBA).
    surf_names : np.ndarray[str] shape(m,)
        Names of label regions.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*contained no data.*')
        surflut = np.loadtxt(surf_lut, usecols=(0,2,3,4,5), dtype="int", comments='#')
        surf_ids = surflut[:,0]
        surf_ctab =  surflut[:,1:5]
        surf_names = np.loadtxt(surf_lut, usecols=(1), dtype="str", comments='#')
        imglut = np.loadtxt(img_lut, usecols=(0,2,3,4,5), dtype="int", comments='#')
        img_ids = imglut[:,0]
        img_names = np.loadtxt(img_lut, usecols=(1), dtype="str", comments='#')
    assert (np.all(img_names == surf_names)), "Label names in the LUTs do not agree!"
    lut = np.zeros((img_labels.max()+1,), dtype = img_labels.dtype)
    lut[img_ids] = surf_ids
    surf_labels = lut[img_labels]
    return surf_labels, surf_ctab, surf_names


def sample_label_to_surface(surf, seg, imglut, surflut, cortex=None, projmm=0.0, dilation=3, radius=None):
    """
    Sample label from image to surface and smooth. Label ID is the value of the label in the segmentation image.

    Parameters
    ----------
    surf : tuple | str
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.array of (n, 3) vertex coordinates and
        surf[1] is the np.array of (m, 3) triangle indices.
        If type is str, read surface from file.
    seg : nibabel.image | str
        Image to sample.
        If type is str, read image from file.
    imglut : str
        Filename for image label look up table.
    surflut : str
        Filename for surface label look up table.
    outaparc : str
        Filename for output surface parcellation.
    cortex : np.ndarray | str
        Filename of cortex label or np.ndarray with cortex indices.
    projmm : float
        Sample projmm mm along the surface vertex normals (default=0).
    radius : float, optional
        If given and if the sample is equal to zero, then consider
        all voxels inside this radius to find a non-zero value.
    """
    if isinstance(cortex, str):
        cortex = fs.read_label(cortex)
    if isinstance(surf, str):
        surf = fs.read_geometry(surf, read_metadata=True)
    if isinstance(seg, str):
        seg = nib.load(seg)
        
    seg_data = np.asarray(seg.dataobj, dtype=int)
    assert np.unique(seg_data).size == 2, "Segmentation image must be binary!"
    #label_value = np.unique(seg_data)[1] dont use this, but just check LUT, so we can also use binary masks
    label_value = np.loadtxt(imglut, usecols=(0,2,3,4,5), dtype="int")[1,0]
    


    # dilate 3D segmentation
    seg_data = binary_dilation(seg_data > 0, iterations=dilation).astype(int)
    seg_data[seg_data == 1] = label_value
    
    
    # get rid of unknown labels first and translate the rest (avoids too much filling
    # later as sampling will search around sample point if label is zero)
    segdata, surfctab, surfnames = replace_labels(seg_data, imglut, surflut)
    # create img with new data (needed by sample img)
    seg2 = nib.MGHImage(segdata, seg.affine, seg.header)
    # sample from image to surface (and search if zero label)
    surfsamples = sample_img(surf, seg2, cortex, projmm, radius)
    
    ## smooth labels on surface
    faces = surf[1]
    nvert = surfsamples.size
    # get Edge matrix (adjacency)
    adjM = get_adjM(faces, nvert)
    # add identity so that each vertex votes in the mode filter below
    adjM = adjM + sparse.eye(adjM.shape[0])
    adjM2 = adjM * adjM
    adjM4 = adjM2 * adjM2
    labels = mode_filter(adjM4, surfsamples)
    labels = mode_filter(adjM2, labels)
    labels = mode_filter(adjM, labels)

    # write annotation
    #fs.write_annot(outaparc, labels, ctab=surfctab, names=surfnames)
    return labels, surfctab, surfnames, adjM


def write_freesurfer_label(filename, vertex_indices, coords=None, values=None, subject='SubjectID', vox2ras='TkReg'):
    """
    Write a FreeSurfer label file.

    Parameters
    ----------
    filename : str
        Path to output label file.
    vertex_indices : np.ndarray
        Array of vertex indices.
    coords : np.ndarray, optional
        Array of (n, 3) vertex coordinates. If None, uses zeros.
    values : np.ndarray, optional
        Array of vertex values. If None, uses 1.0 for all.
    subject : str, optional
        Subject name for the header.
    vox2ras : str, optional
        Coordinate system for the header.
    """
    num_vertices = len(vertex_indices)
    if coords is None:
        coords = np.zeros((num_vertices, 3))
    if values is None:
        values = np.ones(num_vertices)

    with open(filename, 'w') as f:
        f.write(f"#!ascii label  , from subject {subject} vox2ras={vox2ras}\n")
        f.write(f"{num_vertices}\n")
        for i in range(num_vertices):
            idx = vertex_indices[i]
            x, y, z = coords[i]
            val = values[i]
            f.write(f"{idx:d}  {x:f}  {y:f}  {z:f} {val:f}\n")

    logger.info(f"FreeSurfer label file written to: {filename}")


def write_ctab(filename, ctab, names):
    """
    Write a FreeSurfer color table file.

    Parameters
    ----------
    filename : str
        Path to output ctab file.
    ctab : np.ndarray
        Array of (n, 5) color values (R, G, B, A, ID).
    names : list
        List of label names.
    """
    with open(filename, "w") as f:
        for i in range(len(names)):
            name = names[i]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            r, g, b, a, lid = ctab[i]
            f.write(f"{lid:d} {name} {r:d} {g:d} {b:d} {a:d}\n")

    logger.info(f"Color table file written to: {filename}")


def main(insurf: str, inseg: str, incort: str, surflut: str, seglut: str, 
         out_annot: str, projmm: float = 0.0, radius: float = None, 
         to_annot: str = None, dilation: int = 3, report: str = None,
         out_ctab: str = None) -> None:
    """
    Main function to sample labels from segmentation to surface.
    
    Parameters
    ----------
    insurf : str
        Path to input surface
    inseg : str
        Path to input segmentation image
    incort : str
        Path to input cortex label mask
    surflut : str
        FreeSurfer look-up-table for values on surface
    seglut : str
        Look-up-table for values in segmentation image
    out_annot : str
        Path to output annotation or label file
    projmm : float, optional
        Sample along normal at projmm distance (in mm), default 0
    radius : float, optional
        Search around sample location at radius (in mm) for label if 'unknown', default None
    to_annot : str, optional
        Replace annotations in existing annot file
    dilation : int, optional
        Dilation radius for the lesion segmentation mask, default 3
    report : str, optional
        Path to write the surface anatomy report
    out_ctab : str, optional
        Path to write the output color table
    """
    # Load surface for geometry if needed for label file
    surf_geom = fs.read_geometry(insurf, read_metadata=True)
    
    labels, surfctab, surfnames, adjM = sample_label_to_surface(
        surf=surf_geom, 
        seg=inseg, 
        imglut=seglut, 
        surflut=surflut, 
        cortex=incort, 
        projmm=projmm, 
        dilation=dilation, 
        radius=radius
    )
    
    # Track the lesion label value
    # If sampling to a new surface, it's typically the second unique value (background is 0)
    # If appending to existing, it's np.max(existing_labels) + 1
    lesion_mask = None
    
    if to_annot:
        existing_labels, existing_surfctab, existing_surfnames = fs.read_annot(to_annot)
        assert labels.shape == existing_labels.shape, "New labels shape does not match existing labels shape."
        
        labels_in_surf = np.unique(labels)
        
        if len(labels_in_surf) == 2:
            label_value = labels_in_surf[1]
            lesion_mask = (labels == label_value)

            if report:
                logger.info("Generating surface anatomy report...")
                replaced, reduced, adjacent = get_surface_anatomy_info(
                    existing_labels, lesion_mask, adjM
                )
                write_surface_report(report, replaced, reduced, adjacent, existing_surfnames)
        
            if label_value in existing_labels:
                logger.info("Label value already exists, leaving LUT unchanged.")
                surfctab = existing_surfctab
                surfnames = existing_surfnames
            else:
                # append label to surfctab and surfnames
                surfctab = np.loadtxt(surflut, usecols=(0,2,3,4,5), dtype="int")
                surfctab = np.vstack((existing_surfctab, surfctab[-1,[1,2,3,4,0]]))
                surfnames = existing_surfnames + [str.encode(surfnames[-1])]
                
            new_labels = existing_labels
            new_labels[lesion_mask] = np.max(existing_labels) + 1
            labels = new_labels
        elif len(labels_in_surf) > 2:
            raise ValueError("More than one label in surface, cannot append to existing annot file.")
        elif len(labels_in_surf) == 1:
            logger.info('No label found in surface, leaving LUT and outvolume unchanged.')
            if report:
                logger.info("Generating empty surface anatomy report...")
                write_surface_report(report, set(), set(), set(), existing_surfnames)
            labels = existing_labels
            surfctab = existing_surfctab
            surfnames = existing_surfnames
    else:
        # No to_annot provided, check if we have a lesion label
        labels_in_surf = np.unique(labels)
        if len(labels_in_surf) == 2:
            lesion_mask = (labels == labels_in_surf[1])

    if out_annot.endswith('.label'):
        if lesion_mask is None or np.sum(lesion_mask) == 0:
            logger.warning("No lesion label found to write to .label file.")
            # Still write an empty label file or skip? 
            # FreeSurfer usually wants a valid file. Let's write an empty one.
            write_freesurfer_label(out_annot, np.array([], dtype=int))
        else:
            vertex_indices = np.where(lesion_mask)[0]
            # Get coordinates from surface geometry
            coords = surf_geom[0][vertex_indices]
            write_freesurfer_label(out_annot, vertex_indices, coords=coords)
    else:
        fs.write_annot(out_annot, labels, ctab=surfctab, names=surfnames)
        if out_ctab:
            write_ctab(out_ctab, surfctab, surfnames)


def cli():
    # Command Line options are error checking done here
    options = options_parse()

    if options.version:
        import sys
        print(get_version_with_hash())
        sys.exit(0)

    main(
        insurf=options.insurf,
        inseg=options.inseg,
        incort=options.incort,
        surflut=options.surflut,
        seglut=options.seglut,
        out_annot=options.out_annot,
        projmm=options.projmm,
        radius=options.radius,
        to_annot=options.to_annot,
        dilation=options.dilation,
        report=options.report,
        out_ctab=options.out_ctab
    )


if __name__ == "__main__":
    cli()
