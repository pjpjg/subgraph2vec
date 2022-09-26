import os
import pandas as pd
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag

DIR_PATH = os.path.join(os.getcwd(), 'data', 'external')

GENAGE_FILEPATH = os.path.join(DIR_PATH, 'genage_models.csv')
GENE2GO_FILEPATH = os.path.join(DIR_PATH, 'gene2go.txt')
GO_FILEPATH = os.path.join(DIR_PATH, 'go-basic.obo')

OUTPUT_FILEPATH = os.path.join(os.getcwd(), 'data', 'input', 'bp', output_file)


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

# append GO ancestors as a list
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

        # create a dictionary mapping gene terms to go terms and the class
        genAge2go_dict = {}
        # get the geneIDs
        geneIDs = set(genAge2go_org_ns['GeneID'].to_list())
        for geneID in geneIDs:
            # make the geneIDs the keys of the dict with an empty set for GO_IDs
            genAge2go_dict[geneID] = set()
        for index, row in genAge2go_org_ns.iterrows():
            # add the GO_ID directly annotating the geneID
            genAge2go_dict[row['GeneID']].add(row['GO_ID'])
            if VERSION == 'bp':
                # prepare ancestor column
                set_anc = row['ancestors']
                set_anc = row['ancestors'].split()  # this was done because i had saved the file originally and then read it in
                for elem in range(0, len(set_anc)):
                    set_anc[elem] = set_anc[elem][1:-2]     # as above
                set_anc[0] = set_anc[0][1:]                 # as above
                # set_anc = set(set_anc)
                genAge2go_dict[row['GeneID']].update(set_anc)
            else: pass
        print(genAge2go_dict[row['GeneID']])

        # create a dictionary of geneIDs to class
        genAge2class_dict = {}
        for geneID in genAge2go_dict:
            ageing_class = genAge2go_org_ns.query('GeneID=={}'.format(geneID))['longevityInfluence'].iloc[0]
            genAge2class_dict[geneID] = ageing_class

        # now build the binary dataset using the dictionary
        # get all the go terms in the dictionary
        go_terms = set()
        for geneID in genAge2go_dict:
            go_terms.update(genAge2go_dict[geneID])
        try:
            go_terms.remove('')
        except KeyError:
            pass

        # make an empty dataset
        ageing_dataset = pd.DataFrame(columns=go_terms, index=geneIDs)
        # iterate through the cells of df and insert 1 / 0 where appropriate
        for go_term in go_terms:
            for geneID in geneIDs:
                if go_term in genAge2go_dict[geneID]:
                    ageing_dataset.loc[geneID, go_term] = 1
                else:
                    ageing_dataset.loc[geneID, go_term] = 0
        # add class column
        ageing_dataset['class'] = ageing_dataset.index.map(genAge2class_dict)
        # transpose
        ageing_dataset = ageing_dataset.transpose()

        # save the dataset
        filename = '{}_{}.txt'.format(organism, namespace)
        filepath = os.path.join(OUTPUT_FILEPATH, filename)
        ageing_dataset.to_csv(filepath, sep=' ', mode='a')
