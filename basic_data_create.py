import os
import pandas as pd
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag

DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'external')
GENAGE_FILEPATH = os.path.join(DIR_PATH, 'genage_models.csv')
GENE2GO_FILEPATH = os.path.join(DIR_PATH, 'gene2go.txt')
GO_FILEPATH = os.path.join(DIR_PATH, 'go-basic.obo')
OUTPUT_FILEPATH = os.path.join('C:\\Users\\pjgre\\PycharmProjects\\dag2vex\\data\\datasets\\basic_genAge')

# load and prepare genAge dataset
genAge = pd.read_csv(GENAGE_FILEPATH)
genAge.drop(['GenAge ID', 'symbol', 'name', 'avg lifespan change (max obsv)', 'lifespan effect'], axis=1,
                 inplace=True)
genAge.rename(columns={'entrez gene id': 'GeneID', 'longevity influence': 'longevityInfluence'}, inplace=True)
genAge['GeneID'].fillna(0.0, inplace=True)
genAge['GeneID'] = genAge['GeneID'].astype('int64')
genAge['organism'].replace({'Mus musculus': 'MM', 'Drosophila melanogaster': 'DM',
                                'Caenorhabditis elegans': 'CE', 'Saccharomyces cerevisiae': 'SC'}, inplace=True)
genAge['longevityInfluence'].replace({'Anti-Longevity': 0, 'Pro-Longevity': 1}, inplace=True)
genAge.drop(genAge[genAge['longevityInfluence'] == 'Unannotated'].index, inplace=True)
genAge.drop(genAge[genAge['longevityInfluence'] == 'Unclear'].index, inplace=True)

# load and prepare gene2go dataset
gene2go = pd.read_table(GENE2GO_FILEPATH)
gene2go.drop(['#tax_id', 'Evidence', 'Qualifier', 'GO_term', 'PubMed', 'Category'], axis=1, inplace=True)

# load GO
godag = GODag(GO_FILEPATH, optional_attrs={'relationship'})

# merge genAge and gene2go on GeneID
genAge2go = genAge.merge(gene2go, on='GeneID', how='inner')
genAge2go.drop_duplicates(inplace=True)

# append GO namespace
namespaces = []
for go_term in genAge2go['GO_ID']:
    ns = godag[go_term].namespace
    if ns == 'biological_process':
        namespace = 'BP'
    elif ns == 'cellular_component':
        namespace = 'CC'
    elif ns == 'molecular_function':
        namespace = 'MF'
    else: namespace = None
    namespaces.append(namespace)
genAge2go['namespace'] = namespaces

'''
all_ancestors = []
count = 0
for go_term in genAge2go['GO_ID']:
    try:
        gosubdag = GoSubDag([go_term], godag, prt=None)
        ancestors = gosubdag.rcntobj.go2ancestors[go_term]
        all_ancestors.append(ancestors)
    except KeyError:
        all_ancestors.append(set())
    count += 1
    print('{}%'.format((count/len(genAge2go.index)*100)))
genAge2go['ancestors'] = all_ancestors
genAge2go.to_csv(os.path.join(DIR_PATH, 'go2anc.csv'))
'''

# load the dataset holding the ancestors
go2anc = pd.read_csv(os.path.join(DIR_PATH, 'go2anc.csv'), index_col=[0])
for i in go2anc.index:
    ancs = go2anc['ancestors'][i].split()
    for elem in range(0, len(ancs)):
        ancs[elem] = ancs[elem][1:-2]  # as above
    ancs[0] = ancs[0][1:]
    go2anc['ancestors'][i] = ancs

# iterate through for each species and namespace:
for organism in ['DM', 'MM', 'SC', 'CE']:
    genAge2go_org = genAge2go[genAge2go['organism'] == organism]
    for namespace in ['BP', 'CC', 'MF', 'BP-CC', 'BP-MF', 'CC-MF', 'BP-CC-MF']:
        if namespace == 'BP-CC':
            genAge2go_org_ns = genAge2go_org[genAge2go_org['namespace'].isin(['BP', 'CC'])]
        elif namespace == 'BP-MF':
            genAge2go_org_ns = genAge2go_org[genAge2go_org['namespace'].isin(['BP', 'MF'])]
        elif namespace == 'CC-MF':
            genAge2go_org_ns = genAge2go_org[genAge2go_org['namespace'].isin(['CC', 'MF'])]
        elif namespace == 'BP-CC-MF':
            genAge2go_org_ns = genAge2go_org
        else:
            genAge2go_org_ns = genAge2go_org[genAge2go_org['namespace'] == namespace]
        genAge2go_org_ns.drop(['organism', 'namespace'], axis=1, inplace=True)
        # print(organism)
        # print(namespace)
        # print(genAge2go_org_ns.head())

        # append ancestors for each go term
        ancs = []
        for go_term in genAge2go_org_ns['GO_ID']:
            go_ancs = go2anc.loc[go2anc['GO_ID'] == go_term, 'ancestors'].item()
            ancs.append(go_ancs)
        genAge2go_org_ns['ancestors'] = ancs

        # for each gene_ID term, collect go_terms and ancestors, then find non-anc go terms
        geneID_dict = {}
        geneIDs = set(genAge2go_org_ns['GeneID'].to_list())
        for geneID in geneIDs:
            gene_GOs = []
            gene_GO_ancs = set()
            geneID_dict[geneID] = [gene_GOs, gene_GO_ancs]
            for index, row in genAge2go_org_ns.iterrows():
                if row['GeneID'] == geneID:
                    geneID_dict[geneID][0].append(row['GO_ID'])
                    geneID_dict[geneID][1].update(row['ancestors'])
                else: pass

        # now make a dictionary which just maps gene IDs to non-ancestor GO terms
        final_dict = {}
        for geneID in geneIDs:
            final_dict[geneID] = []
        for geneID in geneID_dict:
            for GO_term in geneID_dict[geneID][0]:
                if GO_term not in geneID_dict[geneID][1]:
                    final_dict[geneID].append(GO_term)


        # now the dictionary has been built
        # need to create the binary dataset
        # load original dataset to preserve the order
        original_data_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'genAge',
                                             '{}_{}.txt'.format(organism, namespace))
        with open(original_data_filepath, 'r') as data:
            data_matrix = [list(x.split(",")) for x in data]
        data.close()
        geneIDs = data_matrix[0]
        geneIDs.remove('')
        geneIDs = [int(geneID) for geneID in geneIDs]

        # get the go_terms
        go_terms = set()
        for geneID in final_dict:
            go_terms.update(final_dict[geneID])

        # create a dictionary of geneIDs to class
        genAge2class_dict = {}
        for geneID in final_dict:
            ageing_class = genAge2go_org_ns.query('GeneID=={}'.format(geneID))['longevityInfluence'].iloc[0]
            genAge2class_dict[geneID] = ageing_class

        # make an empty dataset
        ageing_dataset = pd.DataFrame(columns=go_terms, index=geneIDs)
        # iterate through the cells of df and insert 1 / 0 where appropriate
        for go_term in go_terms:
            for geneID in geneIDs:
                if go_term in final_dict[geneID]:
                    ageing_dataset.loc[geneID, go_term] = 1
                else:
                    ageing_dataset.loc[geneID, go_term] = 0
        # add class column
        ageing_dataset['class'] = ageing_dataset.index.map(genAge2class_dict)

        # transpose
        ageing_dataset = ageing_dataset.transpose()

        # save the dataset
        filename = '{}_{}.txt'.format(organism, namespace)
        print(filename)
        filepath = os.path.join(OUTPUT_FILEPATH, filename)
        ageing_dataset.to_csv(filepath, sep=',', mode='a')
        print('saved')
