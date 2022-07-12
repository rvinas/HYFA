from Bio.KEGG import REST
import numpy as np
from pathlib import Path


######################################
# Utilities to retrieve KEGG patways #
######################################

def list_KEGG_human_pathways():
    lines = REST.kegg_list('pathway', 'hsa').readlines()
    symbols = np.array([s.split('\t')[0].split(':')[-1] for s in lines])
    description = np.array([s.split('\t')[1].rstrip() for s in lines])
    return symbols, description


def get_pathway_info(pathway):
    pathway_file = REST.kegg_get(pathway).read()  # query and read each pathway

    # iterate through each KEGG pathway file, keeping track of which section
    # of the file we're in, only read the gene in each pathway
    current_section = None
    gene_symbols = set()
    diseases = set()
    drugs = set()
    for line in pathway_file.rstrip().split('\n'):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == '':
            current_section = section

        if current_section == 'DISEASE':
            disease = line[12:].split(' ')[0]
            diseases.add(disease)
        elif current_section == 'DRUG':
            drug = line[12:].split(' ')[0]
            drugs.add(drug)
        elif current_section == 'GENE':
            try:
                gene_identifiers, gene_description = line[12:].split('; ')
                gene_id, gene_symbol = gene_identifiers.split()
                gene_symbols.add(gene_symbol)
            except ValueError:
                print('WARNING: No gene found in {}'.format(line[12:]))

    return gene_symbols, diseases, drugs


def human_pathway_data(gene_symbols, human_pathways):
    hp = human_pathways
    nb_genes = len(gene_symbols)
    nb_pathways = len(hp)

    genes_p = np.zeros((nb_genes, nb_pathways))

    for i, p in enumerate(hp):
        gs, _, _ = get_pathway_info(p)

        # Store genes of the pathway
        idxs = np.argwhere(np.isin(gene_symbols, list(gs))).flatten()
        genes_p[idxs, i] = 1

    return genes_p


def load_genes_pathway(pathway, gene_symbols, hp_desc, genes_p):
    pathway_idx = np.flatnonzero(np.core.defchararray.find(hp_desc, pathway) != -1)[0]

    # Genes from selected pathway
    genes_from_selected_pathway = genes_p[:, pathway_idx]
    genes_pathway_idxs = np.argwhere(genes_from_selected_pathway)[:, 0]
    gps = gene_symbols[genes_pathway_idxs]
    return gps


##########################################
# Genes belonging to signalling pathways #
##########################################

def _load_pathway_mask(gene_symbols, key):
    print('Loading KEGG pathways information ...')
    hp, hp_desc = list_KEGG_human_pathways()
    genes_p = human_pathway_data(gene_symbols, hp)
    selected_pathways = [p for p in hp_desc if key in p]
    selected_genes = []
    for p in selected_pathways:
        g = load_genes_pathway(p, gene_symbols, hp_desc, genes_p)
        selected_genes.extend(g)
    selected_genes = np.unique(selected_genes)
    gene_mask = np.array([g in selected_genes for g in gene_symbols])
    return gene_mask


def load_pathway_mask(gene_symbols, key='signaling'):
    filename = 'pathways/{key}.npy'
    file = Path(filename)

    if file.is_file():
        with open(filename, 'rb') as f:
            gene_mask = np.load(f)
        return gene_mask

    gene_mask = _load_pathway_mask(gene_symbols, key)

    with open(filename, 'wb') as f:
        np.save(f, np.array(gene_mask))

    print('Selected {}/{} genes associated to "{}"'.format(gene_mask.sum(), len(gene_symbols), key))

    return gene_mask
