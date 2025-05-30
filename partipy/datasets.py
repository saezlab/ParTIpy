import hashlib
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from pybiomart import Dataset
from sklearn.mixture import GaussianMixture

EXPECTED_CHECKSUMS = {
    "hepatocyte_meta.txt": "55aa992aa0473e3ee5598e6da18902d7a11e084f0cd3520668af61469c9067b1",
    "hepatocyte_counts.txt": "20e50fbb9cc81d1a724f437ae6d335518cf6422d4fc0e667386c7a51837f1147",
    "GSE84498%5Fexperimental%5Fdesign.txt.gz": "ca94fce31b850e5fdbf896abd6e9605548f2ac919cca5dc9e0309feeed597ee9",
    "GSE84498%5Fumitab.txt.gz": "3787f1ad635afed6a4169757b71c8c45b7eaa54c69ae2c88ba9d972507b953d8",
    "GSE149859%5Fcolon%5Fprocessed%5Fcounts.txt.gz": "8691de23e46ec7a71e7383763747e041ed28caa44b046bfa17d69d54ac5fd4bf",
}
DATA_PATH = Path("data")


def _compute_partial_sha256(file_path: Path, chunk_size=20 * 1024 * 1024) -> str:
    """Compute a partial SHA256 hash from the start and end of the file."""
    sha256 = hashlib.sha256()
    file_size = file_path.stat().st_size

    with open(file_path, "rb") as f:
        # Read start
        sha256.update(f.read(chunk_size))

        # Read end
        if file_size > chunk_size:
            f.seek(-chunk_size, os.SEEK_END)
            sha256.update(f.read(chunk_size))

    return sha256.hexdigest()


def _file_needs_download(file_path: Path, expected_hash: str) -> bool:
    if not file_path.exists():
        return True
    actual_hash = _compute_partial_sha256(file_path)
    if actual_hash != expected_hash:
        print(f"Checksum mismatch for {file_path.name}: expected {expected_hash}, got {actual_hash}")
        return True
    return False


def load_hepatocyte_data(use_cache: bool = True, data_dir=Path(".") / DATA_PATH, verbose: bool = False):
    """
    Download hepatocyte data from:

    Halpern, K.B., ..., Amit, I., Itzkovitz, S., 2017
    Single-cell spatial reconstruction reveals global division of labour in the mammalian liver
    Nature 542, 352-356
    https://doi.org/10.1038/nature21065

    """
    data_dir.mkdir(exist_ok=True)

    # Define processed cache file
    processed_file = data_dir / "hepatocyte_halpern_processed.h5ad"

    # Check if processed data already exists
    if use_cache and processed_file.exists():
        if verbose:
            print(f"Loading processed data from {processed_file}")
        return anndata.read_h5ad(processed_file)

    # If no cache or cache disabled, download and process from scratch
    file_dicts = {
        "metadata": {
            "filename": "GSE84498%5Fexperimental%5Fdesign.txt.gz",
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84498/suppl/GSE84498%5Fexperimental%5Fdesign.txt.gz",
        },
        "counts": {
            "filename": "GSE84498%5Fumitab.txt.gz",
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84498/suppl/GSE84498%5Fumitab.txt.gz",
        },
    }

    # Download files if needed
    for file_dict in file_dicts.values():
        filepath = data_dir / file_dict["filename"]
        url = file_dict["url"]

        if _file_needs_download(filepath, EXPECTED_CHECKSUMS[file_dict["filename"]]) or not use_cache:
            if verbose:
                print(f"Downloading {url} to {filepath}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            if verbose:
                print(f"Downloaded: {filepath}")
        else:
            if verbose:
                print(f"File already exists, skipping: {filepath}")

    # Process data
    if verbose:
        print("Processing data...")

    # Read metadata and count matrix
    obs = pd.read_csv(data_dir / file_dicts["metadata"]["filename"], sep="\t").set_index("well")
    count_df = pd.read_csv(data_dir / file_dicts["counts"]["filename"], sep="\t").set_index("gene").T.loc[obs.index, :]

    # Construct AnnData
    adata = anndata.AnnData(
        X=count_df.values.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[c.split(";")[0] for c in count_df.columns]),
    )

    # Filter lowly expressed genes
    adata = adata[:, adata.X.sum(axis=0) >= 20].copy()

    # Remove batches of likely non-hepatocytes
    adata = adata[~adata.obs["batch"].isin(["AB630", "AB631"])].copy()

    # Save processed data for faster future loading
    if use_cache:
        if verbose:
            print(f"Saving processed data to {processed_file}")
        adata.write_h5ad(processed_file)

    return adata


def load_hepatocyte_data_2(use_cache=True, data_dir=Path(".") / DATA_PATH, verbose: bool = False):
    """
    Download hepatocyte data from:
     Ben-Moshe, S., ..., Elinav, E., Itzkovitz, S., 2022
     The spatiotemporal program of zonal liver regeneration following acute injury
     Cell Stem Cell 29, 973-989.e10
     https://doi.org/10.1016/j.stem.2022.04.008
    """
    data_dir.mkdir(exist_ok=True)

    # Define processed cache file
    processed_file = data_dir / "hepatocyte_processed.h5ad"

    # Check if processed data already exists
    if use_cache and processed_file.exists():
        if verbose:
            print(f"Loading processed data from {processed_file}")
        return anndata.read_h5ad(processed_file)

    # If no cache or cache disabled, download and process from scratch
    file_dicts = {
        "metadata": {
            "filename": "hepatocyte_meta.txt",
            "url": "https://zenodo.org/records/6035873/files/Single_cell_Meta_data.txt?download=1",
        },
        "counts": {
            "filename": "hepatocyte_counts.txt",
            "url": "https://zenodo.org/records/6035873/files/Single_cell_UMI_COUNT.txt?download=1",
        },
    }

    # Download files if needed
    for file_dict in file_dicts.values():
        filepath = data_dir / file_dict["filename"]
        url = file_dict["url"]

        if _file_needs_download(filepath, EXPECTED_CHECKSUMS[file_dict["filename"]]) or not use_cache:
            if verbose:
                print(f"Downloading {url} to {filepath}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            if verbose:
                print(f"Downloaded: {filepath}")
        else:
            if verbose:
                print(f"File already exists, skipping: {filepath}")

    # Process data
    if verbose:
        print("Processing data...")

    count_tmp = pd.read_csv(data_dir / file_dicts["counts"]["filename"]).set_index("Gene_Name")
    meta_tmp = pd.read_csv(data_dir / file_dicts["metadata"]["filename"])
    meta_tmp = meta_tmp.loc[meta_tmp["Cell_barcode"].isin(count_tmp.columns.to_list())].set_index("Cell_barcode")

    adata = anndata.AnnData(
        X=count_tmp.values.copy().T.astype(np.float32),
        var=pd.DataFrame(index=count_tmp.index.copy()),
        obs=meta_tmp.loc[count_tmp.columns.to_numpy(), :].copy(),
    )
    del count_tmp, meta_tmp

    # Filter data
    adata = adata[(adata.obs["time_point"] == 0) & (adata.obs["cell_type"] == "Hep"), :].copy()
    adata = adata[:, adata.X.sum(axis=0) >= 20].copy()

    # Save processed data for faster future loading
    if use_cache:
        if verbose:
            print(f"Saving processed data to {processed_file}")
        adata.write_h5ad(processed_file)

    return adata


def load_fibroblast_data(use_cache=True, data_dir=Path(".") / DATA_PATH, verbose: bool = False):
    """
    Download fibroblast data from:

    Muhl, L., ..., Betsholtz, C., 2020
    Single-cell analysis uncovers fibroblast heterogeneity and criteria for fibroblast and mural cell identification and discrimination
    Nat Commun 11, 3953
    https://doi.org/10.1038/s41467-020-17740-1
    """
    data_dir.mkdir(exist_ok=True)

    # Define processed cache file
    processed_file = data_dir / "fibroblast_muhl_processed.h5ad"

    # Check if processed data already exists
    if use_cache and processed_file.exists():
        if verbose:
            print(f"Loading processed data from {processed_file}")
        return anndata.read_h5ad(processed_file)

    # If no cache or cache disabled, download and process from scratch
    file_dicts = {
        "counts": {
            "filename": "GSE149859%5Fcolon%5Fprocessed%5Fcounts.txt.gz",
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149859/suppl/GSE149859%5Fcolon%5Fprocessed%5Fcounts.txt.gz",
        },
    }

    # Download files if needed
    for file_dict in file_dicts.values():
        filepath = data_dir / file_dict["filename"]
        url = file_dict["url"]

        if _file_needs_download(filepath, EXPECTED_CHECKSUMS[file_dict["filename"]]) or not use_cache:
            if verbose:
                print(f"Downloading {url} to {filepath}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            if verbose:
                print(f"Downloaded: {filepath}")
        else:
            if verbose:
                print(f"File already exists, skipping: {filepath}")

    # Process data
    if verbose:
        print("Processing data...")

    # translate to gene symbols
    dataset = Dataset(name="mmusculus_gene_ensembl", host="http://www.ensembl.org")
    df = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
    id_to_symbol = dict(zip(df["Gene stable ID"], df["Gene name"], strict=False))

    # prepare the counts
    count_df = pd.read_csv(data_dir / file_dicts["counts"]["filename"], sep="\t")
    count_df = count_df.loc[~count_df.index.str.startswith("ERCC"), :].copy()
    count_df = count_df.loc[count_df.values.sum(axis=1) >= 50, :].copy()
    count_df["gene_symbol"] = count_df.index.map(id_to_symbol)
    count_df = count_df.reset_index(drop=True)
    count_df = count_df.groupby("gene_symbol", as_index=True).sum()

    adata = anndata.AnnData(
        X=count_df.values.T.astype(np.float32),
        obs=pd.DataFrame(index=count_df.columns),
        var=pd.DataFrame(index=count_df.index),
    )
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, mask_var="highly_variable", n_comps=20)
    adata.obs["fibroblast_score"] = adata[:, ["Col1a1", "Pdgfra", "Lum"]].X.mean(axis=1)
    gmm = GaussianMixture(n_components=2, covariance_type="full")
    adata.obs["cluster"] = gmm.fit_predict(adata.obsm["X_pca"])
    fibro_cluster = int(
        adata.obs.groupby("cluster", as_index=False)["fibroblast_score"]
        .mean()
        .nlargest(1, "fibroblast_score")["cluster"]
        .iat[0]
    )
    adata = adata[adata.obs["cluster"] == fibro_cluster, :].copy()
    adata = anndata.AnnData(
        X=adata.layers["counts"].copy(),
        obs=pd.DataFrame(index=adata.obs.index.copy()),
        var=pd.DataFrame(index=adata.var.index.copy()),
    )
    # some filtering
    adata = adata[:, adata.X.sum(axis=0) >= 100].copy()
    adata = adata[adata.X.sum(axis=1) >= 1000, :].copy()

    # Save processed data for faster future loading
    if use_cache:
        if verbose:
            print(f"Saving processed data to {processed_file}")
        adata.write_h5ad(processed_file)

    return adata


def load_ncM_lupus_data(use_cache=True, data_dir=Path(".") / DATA_PATH, verbose: bool = False):
    """
    Download non-classical monocyte data from:

    Perez, R.K., ..., Ye, C.J., 2022
    Single-cell RNA-seq reveals cell type-specific molecular and genetic associations to lupus
    Science 376, eabf1970
    https://doi.org/10.1126/science.abf1970
    https://cellxgene.cziscience.com/collections/436154da-bcf1-4130-9c8b-120ff9a888f2
    """
    import gc

    data_dir.mkdir(exist_ok=True)

    # Define processed cache file
    processed_file = data_dir / "ncM_lupus_processed.h5ad"

    # Check if processed data already exists
    if use_cache and processed_file.exists():
        if verbose:
            print(f"Loading processed data from {processed_file}")
        return anndata.read_h5ad(processed_file)

    # If no cache or cache disabled, download and process from scratch
    if verbose:
        print("Processing data from scratch...")

    # File URL to download
    url = "https://datasets.cellxgene.cziscience.com/4532eea4-24b7-461a-93f5-fe437ee96f0a.h5ad"
    filename = data_dir / os.path.basename(url)

    # Download file if it does not already exist
    if not filename.exists() or not use_cache:
        if verbose:
            print(f"Downloading {url} to {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        if verbose:
            print(f"Downloaded: {filename}")
    else:
        if verbose:
            print(f"File already exists, skipping: {filename}")

    # Process data
    if verbose:
        print("Processing data...")

    adata = anndata.read_h5ad(filename)
    adata.obs["Status"] = adata.obs["disease_state"].map(
        {"managed": "Managed", "na": "Healthy", "flare": "Flare", "treated": "Treated"}
    )
    adata = adata[adata.obs["author_cell_type"] == "ncM", :].copy()  # only consider non-classical monocytes
    adata = adata[adata.obs["Status"] != "Treated", :].copy()  # remove samples with "treated" status

    # remove columns we don't need
    columns_to_drop = [
        "mapped_reference_annotation",
        "cell_type_ontology_term_id",
        "is_primary_data",
        "cell_state",
        "tissue_ontology_term_id",
        "development_stage_ontology_term_id",
        "tissue",
        "organism",
        "tissue_type",
        "suspension_type",
        "organism_ontology_term_id",
        "assay_ontology_term_id",
        "suspension_enriched_cell_types",
        "suspension_uuid",
        "self_reported_ethnicity_ontology_term_id",
        "disease_ontology_term_id",
        "sex_ontology_term_id",
    ]
    # Only drop columns that actually exist
    existing_columns = [col for col in columns_to_drop if col in adata.obs.columns]
    adata.obs.drop(columns=existing_columns, inplace=True)

    # create new index
    adata.obs.index = [
        s.split("-")[0] + "-" + str(len(s.split("-"))) + "-" + str(donor_id)
        for s, donor_id in zip(adata.obs.index, adata.obs["donor_id"].to_list(), strict=False)
    ]

    # remove obsm we don't need (with error handling)
    obsm_to_remove = ["X_pca", "X_umap"]
    for key in obsm_to_remove:
        if key in adata.obsm:
            del adata.obsm[key]

    if hasattr(adata, "uns"):
        del adata.uns
    gc.collect()

    # use the raw counts
    if adata.raw is not None:
        adata.X = adata.raw.X

    # use gene symbols instead of ensembl IDs
    if "feature_name" in adata.var.columns:
        assert len(adata.var["feature_name"]) == len(adata.var["feature_name"].unique())
        adata.var = adata.var.set_index("feature_name")

    # Save processed data for faster future loading
    if use_cache:
        if verbose:
            print(f"Saving processed data to {processed_file}")
        adata.write_h5ad(processed_file)

    return adata
