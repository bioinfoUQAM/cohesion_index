#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Bruno Daigle
# Modifs : Amine Remita
'''
indices de cohesion, calinski et dunn
'''

FONT_SIZE = 90

import sys
import logging
import csv
import numpy as np
from sklearn import metrics
import seaborn as sns

from matplotlib import rcParams
# rcParams['font.size'] = 50
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman','Times']
rcParams['xtick.labelsize'] = FONT_SIZE
rcParams['ytick.labelsize'] = FONT_SIZE

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

################################################################################

# plt.rcParams['font.size'] = 24

################################################################################

def minInterDistance(ligneDistances, labels, i) :
    mask = labels != labels[i]
    mi = np.amin(ligneDistances[mask])
    return mi

def cohesionItem(ligneDistances, Min, labels, i) :
    mask = labels == labels[i]
    mask[i] = False #pas de comparaison avec soi-même
    ci = ligneDistances[mask] <= Min[i]
    if len(ci) == 0 :
        return 0.5
    return np.sum(ci) / float(len(ci))
    
def cohesion_samples(XouD, labels, metric='euclidean') :
    n = labels.shape[0]
    X = metrics.pairwise_distances(XouD, metric=metric)
    Min = np.array([minInterDistance(X[i], labels, i)
                  for i in range(n)])
    coh = np.array([cohesionItem(X[i], Min, labels, i)
                  for i in range(n)])
    return coh    

def cohesion_score(X, labels, metric='euclidean') :
    return np.mean(cohesion_samples(X, labels, metric=metric))    

################################################################################
# calcul de la somme des carres des dissimilarites entre les elements de
# la liste elements selon les valeurs de X
# on assume une matrice symetrique, i.e. d(i,j)=d(j,i) et d(i,i) == 0
# 
def sommeCarres(X) :
    return np.sum(X ** 2)
    # nb=len(X)
    # somme = 0.0
    # for i in range(0, nb - 1) :
        # for j in range(i + 1, nb) :
            # somme += X[i,j] ** 2
    # return somme * 2 #symetrique

def calinski(X, labels) :
    n = labels.shape[0]
    WssTout = sommeCarres(X)/n
    groupes, tailles = np.unique(labels, return_counts=True)
    k = len(groupes)
    
    WssGroupes = 0.0 
    for i in range(0, k) :
        #print groupes[i]
        mask = labels == groupes[i]
        w = sommeCarres(X[:, mask][mask, :]) / tailles[i]
        WssGroupes += w

    Bss = WssTout - WssGroupes
    ch = (Bss * (n - k)) / (WssGroupes * (k - 1))

    return ch    

################################################################################
def dunn(X, labels) :
    n = labels.shape[0]
    groupes = np.unique(labels, return_counts=False)
    k = len(groupes)
    diam = []
    minInter = []

    for i in range(0, k) :
        maski = labels == groupes[i]
        diam.append(np.max(X[:, maski][maski, :]))
        for j in range(i+1, k) :
            maskj = labels == groupes[j]
            minInter.append(np.min(X[maski, :][:, maskj]))
    d = min(minInter) / max(diam)
    return d    

################################################################################
def plot_indice(X, silh_s, silhouette_avg, n_clusters, 
              cluster_labels, nomClasse, cList ,xLim1, xLim2, 
              titre, xLabel, yLabel, outFig):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 30)
    
    ###################################
    # The silhouette coefficient can range from -1 to 1
    ax.set_xlim([xLim1, xLim2])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    #ax.set_xticks(np.arange(-1.0, 1.0, 0.2))
    y_lower = 10
    
    # colors = cm.magma(cluster_labels.astype(float) / n_clusters)
    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            silh_s[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        # color = cm.spectral(float(i) / n_clusters)
        color = cList[i]
        
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        # ax.text(xLim1 + 0.1, y_lower + 0.2 * size_cluster_i,
                 # nomClasse[i], fontdict={'fontsize':(FONT_SIZE-10)})
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    # ax.set_title(titre)
    ax.set_xlabel(xLabel, fontsize=FONT_SIZE)
    ax.set_ylabel(yLabel, fontsize=FONT_SIZE)
    
    # The vertical line for average silhoutte score of all the values
    # label="moyenne {:0.2f}".format(silhouette_avg)
    # leg = ax.axvline(x=silhouette_avg, color="red", linestyle="--",)
    # ax.legend(loc="lower right", fontsize='small', framealpha=0.2)
    
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    
    # plt.show()
    fig.savefig(outFig)

################################################################################
def plot_clusters(n_clusters, cluster_labels, tabLims, mds, cList, titre, xLabel, yLabel, outFig):

    fig, ax = plt.subplots()
    fig.set_size_inches(31, 31)
    
    # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    # print (colors)
    colors = [cList[x] for x in cluster_labels]
    #ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #            c=colors)
    
    if len(tabLims) > 0 :
      ax.set_xlim([tabLims[0][0], tabLims[0][1]])
      ax.set_ylim([tabLims[1][0], tabLims[1][1]])
    
    ax.scatter(mds[:, 0], mds[:, 1], marker='o', edgecolors='face',
                s=2500, c=colors, alpha=0.4, linewidths=0)
    
    # Labeling the clusters
    #centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    #ax.scatter(centers[:, 0], centers[:, 1],
    #            marker='o', c="white", alpha=1, s=200)
    
    #for i, c in enumerate(centers):
    #    ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
    
    ax.set_title(titre)
    ax.set_xlabel(xLabel, fontsize=FONT_SIZE)
    ax.set_ylabel(yLabel, fontsize=FONT_SIZE)
    
    # ax.set_yticks([])  # Clear the yaxis ticks
    # ax.set_xticks([])  # Clear the xaxis ticks

    # plt.show()
    fig.savefig(outFig)
    

################################################################################
################ affichage graphique
# adapté de http://scikit-learn.org/stable/auto_examples/cluster/
#                                       plot_kmeans_silhouette_analysis.html
def plot_indices(X, silh_s, silhouette_avg, cohe_s, cohesion_avg,
                 n_clusters, cluster_labels, nomClasse, titre, mds):
    # Create a subplot with 1 row and 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(28, 14) # ???

    ###################################
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1 to 1
    ax1.set_xlim([np.amin(silh_s) - 0.1, 1.05])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    #ax1.set_xticks(np.arange(-1.0, 1.0, 0.2))
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            silh_s[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(silhouette_avg + 0.05, y_lower + 0.5 * size_cluster_i,
                 nomClasse[i])
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("Silhouette")
    ax1.set_xlabel("indice silhouette")
    ax1.set_ylabel("clusters")
    
    # The vertical line for average silhoutte score of all the values
    leg = ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                      label="moyenne {:0.2f}".format(silhouette_avg))
    ax1.legend(loc="lower right", fontsize='small', framealpha=0.2)
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    ###################################
    # The 2nd subplot is the cohesion plot
    # The cohesion coefficient can range from 0 to 1
    ax3.set_xlim([-0.05, 1.05])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax3.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    #ax3.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #ax3.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            cohe_s[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.spectral(float(i) / n_clusters)
        ax3.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax3.text(cohesion_avg + 0.05, y_lower + 0.5 * size_cluster_i,
                 nomClasse[i])
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax3.set_title("Cohesion")
    ax3.set_xlabel("indice de cohesion")
    ax3.set_ylabel("clusters")
    
    # The vertical line for average silhoutte score of all the values
    leg = ax3.axvline(x=cohesion_avg, color="red", linestyle="--",
                      label="moyenne {:0.2f}".format(cohesion_avg))
    ax3.legend(loc="lower right", fontsize='small', framealpha=0.2)
    ax3.set_yticks([])  # Clear the yaxis labels / ticks
    #ax3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    ###################################
    # 3rd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    #ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #            c=colors)
    
    ax2.scatter(mds[:, 0], mds[:, 1], marker='o', edgecolors='face',
                s=40, c=colors)
    # Labeling the clusters
    #centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    #ax2.scatter(centers[:, 0], centers[:, 1],
    #            marker='o', c="white", alpha=1, s=200)
    
    #for i, c in enumerate(centers):
    #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
    
    ax2.set_title("Multidimensional Scaling")
    ax2.set_xlabel("composante 1")
    ax2.set_ylabel("composante 2")
    ax2.set_yticks([])  # Clear the yaxis ticks
    ax2.set_xticks([])  # Clear the xaxis ticks
    plt.suptitle(titre, fontsize=14, fontweight='bold')
    if matplotlib.get_backend() == 'Qt4Agg':
        # avec autre backend, il faudrait trouver une autre recette pour
        # maximiser la fenêtre
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()  
    plt.show()
