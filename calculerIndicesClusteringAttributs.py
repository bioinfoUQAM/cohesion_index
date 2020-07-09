#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Bruno Daigle
# Modifs : Amine Remita
'''
Usage: {} fichierExemple-Attributs metrique outResultFile outSilhFig outCohesFig outClustFig

Calcule les indices de validite de clustering silhouette, cohesion, Calinski et
Dunn.

fichierExemple-Attributs :
    nom du fichier d'exemples-attributs csv au format suivant:
        1ère ligne:       entête des colonnes
        1ère colonne:     identifiant de l'exemple
        dernière colonne: classe de l'exemple
        autres colonnes:  attributs numériques de l'exemple
    
    Par exemple, le fichier 'ex.csv' contenant:
        SEQID,RFLP_CUT_AatII,RFLP_CUT_AbsI,RFLP_CUT_AccI,RFLP_CUT_AclI,Genre
        AF349909,1,0,4,0,ALPHA_HPV
        AF534061,0,0,13,4,ALPHA_HPV
        AJ620210,1,0,5,3,GAMMA_HPV
        AY330621,0,0,9,1,ALPHA_HPV
        AY382778,0,0,6,2,BETA_HPV

metrique :
    mesure de distance à utiliser (voir scipy.spatial.distance.pdist), parmi:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'

Le résultat est produit sur stdout en csv au format suivant:
    1ère ligne:       entête des colonnes
    1ère colonne:     identifiant sur lequel porte les indices
    colonnes 2 à 5:   dans l'ordre les indices silhouette, cohesion, calinski,
                      dunn
    colonne 6:        classe de l'exemple
    colonne 7:        type de classe (dernier champs de la ligne d'entête du
                      fichier d'input)
    dernière colonne: indique si les indices portent sur les exemples
                      individuels, les groupes ou l'ensemble global. Dans ce
                      dernier cas, les colonnes 1 et 6 contiennet plutôt le nom
                      du fichier d'input

Exemple de résultats pour le fichier 'ex.csv':
    id,silhouette,cohesion,calinski,dunn,classe,type_classe,type_resultat
    AF349909,-0.602534,0.000000,,,ALPHA_HPV,Genre,individuel
    AF534061,-0.022771,0.500000,,,ALPHA_HPV,Genre,individuel
    AJ620210,0.000000,0.500000,,,GAMMA_HPV,Genre,individuel
    AY330621,-0.379712,0.000000,,,ALPHA_HPV,Genre,individuel
    AY382778,0.000000,0.500000,,,BETA_HPV,Genre,individuel
    ALPHA_HPV,-0.335006,0.166667,,,ALPHA_HPV,Genre,groupe
    GAMMA_HPV,0.000000,0.500000,,,GAMMA_HPV,Genre,groupe
    BETA_HPV,0.000000,0.500000,,,BETA_HPV,Genre,groupe
    ex.csv,-0.201003,0.300000,0.288000,0.174964,ex.csv,Genre,global

Note:
    - calinski et dunn ne s'appliquent qu'au global
'''
import numpy as np
from sklearn import metrics
from sklearn import manifold
import indicesClustering
import scipy
import sys
import os
import csv

metriques = [
    'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'matching',
    'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    ]
if (len(sys.argv) != 9 and len(sys.argv) != 13) or sys.argv[2] not in metriques:
    # print(__doc__ .format(sys.argv[0]))
    print ('sortie 1')
    sys.exit()

inFile         = sys.argv[1]
metrique       = sys.argv[2]
colorFile      = sys.argv[3]
outResultFile  = sys.argv[4]
outSilhFig     = sys.argv[5]
outCohesFig    = sys.argv[6]
outClustFig    = sys.argv[7]
clustPlotLims  = sys.argv[8]

if clustPlotLims == '1':
  if len(sys.argv) != 13 :
    # print(__doc__ .format(sys.argv[0]))
    print ('sortie 2')
    sys.exit()
  xL1            = int (sys.argv[9])
  xL2            = int (sys.argv[10])
  yL1            = int (sys.argv[11])
  yL2            = int (sys.argv[12])


################ lire le csv exemples/attrinuts
f = open(inFile, 'r')
reader = csv.reader(f)

# entete
entete = next(reader)
nbChamps = len(entete)
assert nbChamps >= 3, "Nombre de champs insuffisant dans {}".format(sys.argv[1])
type_de_classe = entete[-1]

nomClasse = [] # nom de la classe
ids = []       # nom de chaque exemple
classes = []   # numéro de la classe de chaque exemple
data = []      # les données liste de liste des attributs de chaque exemple


for ligne in reader:
    if len(ligne) == 0: continue
    assert len(ligne) == nbChamps, "ligne invalide: {}".format(ligne)
    ids.append(ligne[0])
    if ligne[-1] not in nomClasse :
        nomClasse.append(ligne[-1])
    classes.append(nomClasse.index(ligne[-1]))
    data.append(ligne[1:nbChamps-1])
    
f.close()

# Ouverture du fichier output
fout = open(outResultFile, 'w')

################ calculer les indices de chaque exemple
# data = np.asarray(data, dtype = float) # pas besoin, csvreader le fait déjà
# matrice de distance
d = scipy.spatial.distance.pdist(data, metric=metrique)
d = scipy.spatial.distance.squareform(d)
classes = np.asarray(classes)

# les indices
silh_s = metrics.silhouette_samples(d, classes, metric='precomputed')
#silh = metrics.silhouette_score(d, classes, metric='precomputed')
cohe_s = indicesClustering.cohesion_samples(d, classes, metric='precomputed')
#cohe = indicesClustering.cohesion_score(d, classes)
cal = indicesClustering.calinski(d, classes)
dun = indicesClustering.dunn(d, classes)


################ les résultats
# entête
fout.write("id,silhouette,cohesion,calinski,dunn,classe,type_classe,type_resultat\n")

# indices de chaque exemple
for i in range(len(ids)):
    fout.write("{},{:4f},{:4f},,,{},{},individuel\n".format(ids[i], silh_s[i],
                                                       cohe_s[i],
                                                       nomClasse[classes[i]],
                                                       type_de_classe))

# indices de chaque classe (c'est la moyenne des indices des exemples de la
# classe)
for i in range(len(nomClasse)):
    mask = classes == i
    fout.write("{0},{1:4f},{2:4f},,,{0},{3},groupe\n".format(nomClasse[i],
                                                    np.mean(silh_s[mask]),
                                                    np.mean(cohe_s[mask]),
                                                    type_de_classe))

# indices globaux
silhouette_avg = np.mean(silh_s)
cohesion_avg = np.mean(cohe_s)
fout.write("{},{:4f},{:4f},{:4f},{:4f},{},{},global\n".format(sys.argv[1],
                                                   silhouette_avg,
                                                   cohesion_avg, cal, dun,
                                                   sys.argv[1], type_de_classe))
fout.close()

################ affichage graphique

colorList = [line.strip() for line in open(colorFile, 'r')]
# calculer le multidimensional scaling
mdsEng = manifold.MDS(n_components=2, metric=True, max_iter=300, eps=1e-6,
                      dissimilarity="precomputed", random_state=1)  #n_jobs=-1
mds = mdsEng.fit(d).embedding_

# titre = "Indices silhouette et cohesion: {} clusters\n" \
#        "fichier: {}, métrique: {}".format(len(nomClasse), sys.argv[1],
#                                           metrique)
# indicesClustering.plot_indices(d, silh_s, silhouette_avg, cohe_s,
                               # cohesion_avg, len(nomClasse), classes,
                               # nomClasse, titre, mds)

# silXLim1  = np.amin(silh_s) - 0.1
silXLim1  = -0.4
silXLim2  = 1.05
silTitre  = ""
silXlabel = "Silhouette index"
silYlabel = "Classes"
indicesClustering.plot_indice(d, silh_s, silhouette_avg, len(nomClasse), 
                            classes, nomClasse, colorList, silXLim1, silXLim2, 
                            silTitre, silXlabel, silYlabel, outSilhFig)
                            
cohXLim1  = -0.05
cohXLim2  = 1.05
cohTitre  = ""
cohXlabel = "Cohesion index"
cohYlabel = "Classes"
indicesClustering.plot_indice(d, cohe_s, cohesion_avg, len(nomClasse), 
                            classes, nomClasse, colorList, cohXLim1, cohXLim2, 
                            cohTitre, cohXlabel, cohYlabel, outCohesFig)

mdsTitle = ""
mdsXlabel = "Component 1"
mdsYlabel = "Component 2"
tabLims = []

if clustPlotLims == '1':
  tabLims = [[xL1,xL2],[yL1,yL2]]

indicesClustering.plot_clusters(len(nomClasse), classes, tabLims, mds, colorList, mdsTitle, mdsXlabel, mdsYlabel, outClustFig)
