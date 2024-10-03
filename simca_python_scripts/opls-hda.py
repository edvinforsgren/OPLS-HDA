"""
Code related to:
Title: OPLS-based Multiclass Classification and Data-Driven Inter-Class Relationship Discovery
Description: Orthogonal Partial Least Squares-Hierarchical Discriminant Analysis (OPLS-HDA)
OPLS-HDA integrates Hierarchical Cluster Analysis (HCA) with the OPLS-DA
framework to create a decision tree, addressing multiclass classification challenges and 
providing intuitive visualization of inter-class relationships. To avoid overfitting and
ensure reliable predictions, we use cross-validation during model building.

Authors:
- Edvin Forsgren (edvin.forsgren@gmail.com)
- Pär Jonsson (paer.jonsson@sartorius.com)

Last edit: 2024-10-02

MIT License

Copyright (c) 2024 Edvin Forsgren, Pär Jonsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import umetrics
import umpypkg
import copy
import seaborn as sns
import numpy as np
from sklearn.utils import shuffle
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from scipy.cluster import hierarchy
import scipy.stats as st
from scipy.stats import t
import numpy.matlib
from scipy.stats.distributions import chi2
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score

class Auto_fit:
    def __init__(self, modelNUM,AFoption,AFoption2=None):
        project = umetrics.SimcaApp.get_active_project()
        obs=project.get_model_info(modelNUM).observations
        var=project.get_model_info(modelNUM).Xvariables
        max_ortho=np.floor(min(obs-2,var,30)/2)
        if AFoption=='af':
            comp=project.fit_model(modelNUM)
            if sum(comp)==0:
                comp=project.fit_model(modelNUM,numcomp=1, numorthX=0)
            if (sum(comp)-1)>=max_ortho:
                comp=project.fit_model(modelNUM,numcomp=1, numorthX=int(max_ortho))
        elif AFoption=='max_q2':
            q2=np.zeros((int(round(max_ortho)),1))
            for oc in range(0,int(round(max_ortho))):
                project.fit_model(modelNUM,numcomp=1, numorthX=oc)
                q2[oc]=project.data_builder().create('Q2cum', project.get_active_model()).matrix()[0][0]
            a=np.where(q2==np.max(q2,0))
            oc2=int(round(min(a[0])))
            comp=project.fit_model(modelNUM,numcomp=1, numorthX=oc2)
        elif AFoption=='min_cv':
            pcva=np.ones((int(round(max_ortho)),1))
            for oc in range(0,int(round(max_ortho))):
                comp=project.fit_model(modelNUM,numcomp=1, numorthX=oc)
                Y=np.round(np.array(project.data_builder().create("YVar",model=project.get_active_model()).matrix()))
                YPredcv=np.array(project.data_builder().create("YPredcv",model=project.get_active_model()).matrix())
                SSres=np.sum((Y[0,:]-YPredcv[0,:])**2)
                SStot=np.sum((Y[0,:]-np.mean(Y[0,:]))**2)
                SSreg=SStot-SSres
                DFreg=2*sum(comp)
                DFres=obs-DFreg-1
                F=(SSreg/DFreg)/(SSres/DFres)
                F=np.float64(F)
                p = st.f.cdf(1/F, DFres, DFreg)
                if SSreg<=0:
                    p=1
                pcva[oc]=p
            a=np.where(pcva==np.min(pcva,0))
            oc2=int(round(min(a[0])))
            comp=project.fit_model(modelNUM,numcomp=1, numorthX=oc2)
        elif AFoption=='choose':
            if AFoption2>max_ortho:
                AFoption2=int(max_ortho)
            comp=project.fit_model(modelNUM,numcomp=1, numorthX=AFoption2)

def one_vs_one_calc(AFoption, scale_option, dataset_name, excl_from_training_set, train_df, class_col, excl_test_data, n_components=None):
    project = umetrics.SimcaApp.get_active_project()
    base_workset = project.create_workset(dataset_name)
    base_workset.set_type(umetrics.simca.modeltype.oplsDa)
    excl_indices = list(set(base_workset.get_observations()).difference([i for i in train_df.index.tolist()]).copy())

    excl_indices = [base_workset.get_observations().index(element) for element in excl_indices]
    if not excl_test_data:
        base_workset.exclude_obs(excl_from_training_set)
    if excl_indices:
        base_workset.exclude_obs(excl_indices)
    u_class_names = train_df[class_col].unique().tolist()
    for i, class_name in enumerate(u_class_names):
        c_index = train_df[train_df[class_col] == class_name].index
        base_workset.set_obs_class(c_index, i+1, class_name)
    classes = np.take(base_workset.get_obs_class_phase_numbers(), base_workset.get_included_obs(), axis=0)
    D=np.zeros((np.size(np.unique(classes)),np.size(np.unique(classes))))
    Dnames=[]
    unique_classes = np.unique(classes)
    models = []
    for i, uci in enumerate(unique_classes):
        for j, ucj in enumerate(unique_classes):
            if i<j:
                workset = project.create_workset(dataset_name)
                workset.set_type(umetrics.simca.modeltype.oplsDa)
                # Exclude data
                if not excl_test_data:
                    workset.exclude_obs(excl_from_training_set)
                if excl_indices:
                    workset.exclude_obs(excl_indices)
                # Set class belongings
                for ci, class_name in enumerate(u_class_names):
                    c_index = train_df[train_df[class_col] == class_name].index
                    workset.set_obs_class(c_index, ci+1, class_name)
                # Exclude all classes that are not in this iteration
                excl=np.intersect1d(np.where(classes!=uci),np.where(classes!=ucj)).tolist()
                excl_from_model = [workset.get_included_obs()[x] for x in excl]
                workset.exclude_obs(excl_from_model)
                # Set scaling
                if scale_option == 'uv':
                    workset.set_variable_scale_type([], umetrics.simca.scaletype.uv)
                elif scale_option == 'ctr':
                    workset.set_variable_scale_type([], umetrics.simca.scaletype.ctr)
                elif scale_option == 'none':
                    workset.set_variable_scale_type([], umetrics.simca.scaletype.none)
                model = workset.create_model()
                # fit model
                if n_components:
                    Auto_fit(model[0], AFoption, n_components)
                else:
                    Auto_fit(model[0], AFoption)
                comp=[project.get_model_info(model[0]).components,project.get_model_info(model[0]).orthogonalinX]
                Mname=project.get_model_info(model[0]).classnames
                if comp[0]>0:
                    Y=np.round(np.array(project.data_builder().create("YVar",model=model[0]).matrix()))
                    YPredcv=np.array(project.data_builder().create("YPredcv",model=model[0]).matrix())
                    
                    m0=np.mean(YPredcv[0,np.where((Y[0,:])==np.min(Y[0,:]))])
                    m1=np.mean(YPredcv[0,np.where((Y[0,:])==np.max(Y[0,:]))])
                    n0=np.size(YPredcv[0,np.where((Y[0,:])==np.min(Y[0,:]))])
                    n1=np.size(YPredcv[0,np.where((Y[0,:])==np.max(Y[0,:]))])
                    ss0=np.sum((YPredcv[0,np.where((Y[0,:])==np.min(Y[0,:]))]-m0)**2)
                    ss1=np.sum((YPredcv[0,np.where((Y[0,:])==np.max(Y[0,:]))]-m1)**2)
                    ss=ss1+ss0 
                    stdpool=np.sqrt(ss/(n0+n1-2))
                    d=(m1-m0)/stdpool # Cohen's d
                else:
                    d=sum(np.random.rand(1)/1e6)
                if d<=0:
                    d=sum(np.random.rand(1)/1e6)
                # Fill the distance matrix with calculated Cohen's d
                D[i,j]=d 
                D[j,i]=d
                # Append the names of the classes 
                if j==i+1:
                    Dnames.append(Mname[0])
                str1=["One vs One:",Mname[0], " vs. ", Mname[1]," Dist(",str(i) ,",",str(j), ")=",np.array2string(d, formatter={'float_kind':lambda d: "%.3f" % d})] 
                s = ''.join(str1)
                project.get_model_info(project.get_active_model()).set_description(s)
                models.append(model[0])
                yield D, Dnames, models
    Dnames.append(Mname[1])  # Names ordered as in D
    yield D, Dnames, models
    df = pd.DataFrame(columns=Dnames, index=Dnames, data=D)
    p = umetrics.SimcaApp.get_active_project()
    try:
        p.delete_dataset('Distances')
    except:
        pass
    imp = umetrics.impdata.ImportData()
    imp.set(df)
    p.create_dataset(imp, str('Distances'))


def bin_models_calc(AFoption, scale_option, linked_data, dataset_name, excl_from_training_set, train_df, class_col, Dnames, excl_test_data, n_components=None):
    project = umetrics.SimcaApp.get_active_project()
    base_workset = project.create_workset(dataset_name)
    base_workset.set_type(umetrics.simca.modeltype.oplsDa)
    excl_indices = list(set(base_workset.get_observations()).difference([i for i in train_df.index.tolist()]).copy())
    excl_indices = [base_workset.get_observations().index(element) for element in excl_indices]
    if not excl_test_data:
        base_workset.exclude_obs(excl_from_training_set)
    if excl_indices:
        base_workset.exclude_obs(excl_indices)
    u_class_names = train_df[class_col].unique().tolist()
    for i, class_name in enumerate(u_class_names):
        c_index = train_df[train_df[class_col] == class_name].index
        base_workset.set_obs_class(c_index, i+1, class_name)
    classes = np.take(base_workset.get_obs_class_phase_numbers(), base_workset.get_included_obs(), axis=0)
    unique_classes = np.unique(classes)
    # This creates the mapping of hierarchies between classes
    C=np.array([i for i in range(len(unique_classes))])
    C=np.matlib.repmat(C, len(unique_classes), 1)
    C=C.transpose()
    for i in range(0, len(unique_classes)-1):
        z=linked_data[i,0:2]
        n0=np.where(C[:,i]==z[0])
        n1=np.where(C[:,i]==z[1])
        k=np.max(C,axis=None)+1
        C[n0,i+1]=k
        C[n1,i+1]=k
        if i+2<=np.shape(C)[1]-1:
            C[:,i+2]=C[:,i+1]
    C=np.fliplr(C)
    C2=C*0
    dn1 = dendrogram(linked_data,ax=None, no_plot=True)
    leaves = dn1['leaves']
    # CALCULATES BINARY CLASSIFIERS using C as a basis
    for i in range(0, len(unique_classes)-1):
        poss=np.unique(np.where(C[:,i]!=C[:,i+1]))
        poss1_before=np.unique(np.where(C[poss,i+1]==np.max(C[poss,i+1])))
        poss2_before=np.unique(np.where(C[poss,i+1]==np.min(C[poss,i+1])))

        if min([leaves.index(poss[p]) for p in poss1_before]) < min([leaves.index(poss[p]) for p in poss2_before]):
            poss1=poss[poss1_before]
            poss2=poss[poss2_before]
        else:
            poss1=poss[poss2_before]
            poss2=poss[poss1_before]
        C2[poss1,i]=1
        C2[poss2,i]=2
        incl=[]
        class1=[]
        class2=[]
        
        # Below we make sure to include the correct classes in the model
        for p in poss1:
            incl=np.union1d(incl, np.where(classes==unique_classes[p]))
            class1=np.union1d(class1, np.where(classes==unique_classes[p]))
        for p in poss2:
            incl=np.union1d(incl, np.where(classes==unique_classes[p]))
            class2=np.union1d(class2, np.where(classes==unique_classes[p]))
       
        workset = project.create_workset(dataset_name)
        excl=np.setdiff1d(range(0,len(classes)),incl).tolist()
        class1=np.intersect1d(range(0,len(classes)),np.round(class1)).tolist()
        class2=np.intersect1d(range(0,len(classes)),np.round(class2)).tolist()
        excl=list(map(int, excl))
        if not excl_test_data:
            workset.exclude_obs(excl_from_training_set)
        if excl_indices:
            workset.exclude_obs(excl_indices)
        excl_from_model = [workset.get_included_obs()[x] for x in excl]
        class1=list(map(int, class1))
        class2=list(map(int, class2))
        class1 = [workset.get_included_obs()[x] for x in class1]
        class2 = [workset.get_included_obs()[x] for x in class2]
        # Here we assign the classes in each group to class 1 and 2
        workset.set_obs_class(class1,1,'class1')
        workset.set_obs_class(class2,2,'class2')
        workset.exclude_obs(excl_from_model)
        # Set scaling
        workset.set_type(umetrics.simca.modeltype.oplsDa)
        if scale_option == 'uv':
            workset.set_variable_scale_type([], umetrics.simca.scaletype.uv)
        elif scale_option == 'ctr':
            workset.set_variable_scale_type([], umetrics.simca.scaletype.ctr)
        elif scale_option == 'none':
            workset.set_variable_scale_type([], umetrics.simca.scaletype.none)
        model = workset.create_model()
        # Fit model
        if n_components:
            Auto_fit(model[0],AFoption, n_components)
        else:
            Auto_fit(model[0],AFoption)
        # Extract which split the model is in to set the model name
        if len(C[poss1,:])==1:
            strC1=Dnames[int(np.squeeze(poss1))]
        if len(C[poss1,:])>1:
            temp=np.std(np.array(C[poss1,:]),axis=0)
            a=np.min(np.where(temp>0))
            strC1=''.join(["Split #",str(a)] )
        if len(C[poss2,:])==1:
            strC2=Dnames[int(np.squeeze(poss2))]
        if len(C[poss2,:])>1:
            temp=np.std(np.array(C[poss2,:]),axis=0)
            a=np.min(np.where(temp>0))
            strC2=''.join(["Split #",str(a)] )
        # Get class label prediction based on condition from class pred (YPredCV)
        y_preds = project.data_builder().create("YPredcv", model=model[0]).matrix()[0]
        y_preds = np.array(y_preds) > 0.5
        y_true = project.data_builder().create("YVar", model=model[0]).matrix()[0]
        y_true = np.array(y_true) > 0.5
        acc = np.round(np.sum(y_preds == y_true)/len(y_true)*100, 2)
        str1=["Binary Classifier.","Split #",str(i+1)," ==> ",strC1," vs ",strC2, " - CV-Accuracy: ", str(acc), "%"] 
        s = ''.join(str1)    
        project.get_model_info(model[0]).set_description(s)
        yield model, i


def concat_dfs_from_simca():
    # This function creates pandas dataframes from the projects datasets
    project = umetrics.SimcaApp.get_active_project()
    dataset_infos = project.get_dataset_infos()
    datasets = []
    dataset_names = []
    for b in dataset_infos:
        if b.type == 1:
            datasets.append(b.ID)
            dataset_names.append(b.name)
    df_dict = {}
    for dataset, name in zip(datasets, dataset_names):
        val_ids = project.data_builder().create("ObsID", dataset=dataset).get_value_ids()
        workset = project.create_workset(dataset)
        col_names = val_ids.get_id_names()
        col_names = ["Metadata_" + col.split('Obs ID (')[1][:-1] for col in col_names]
        col_data = []
        df = pd.DataFrame(columns=col_names)
        for i, col in enumerate(col_names):
            df[col] = val_ids.get_names(i+1)
        df.set_index('Metadata_Primary', inplace=True)
        df = pd.concat([df, workset.get_data_collection().as_dataframe()], axis=1)
        df_dict.update({f"{name}": df})
    return(df_dict)

def confidence_ellipse(x, y, ax, **kwargs):
    """
    Create a plot of the covariation confidence ellipse op `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data

    Returns
    -------
    float: the Pearson Correlation Coefficient for `x` and `y`.

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties

    author : Carsten Schelp
    license: GNU General Public License v3.0 (https://github.com/CarstenSchelp/CarstenSchelp.github.io/blob/master/LICENSE)
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    n_std=np.sqrt(chi2.ppf(0.95,df=2))#scipy.stats.f.ppf(0.95, 4, N-4)

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1,1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    # calculating the stdandarddeviation of x from  the squareroot of the variance
    # np.sqrt(cov[0, 0])
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the stdandarddeviation of y from  the squareroot of the variance
    # np.sqrt(cov[1, 1])
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
        
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

def plot_scores(model, ax, fig, df, class_col):
    project = umetrics.SimcaApp.get_active_project()
    m_info = project.get_model_info(model)
    ok='yes2d'
    if m_info.components!=1:
        ok='no'
    elif m_info.orthogonalinX==0:
        ok='yes1d'

    builder = project.data_builder()
    workset = project.new_as_workset(model)
    classes = np.take(workset.get_obs_class_phase_numbers(), workset.get_included_obs(), axis=0)
    ws_incl = project.data_builder().create("ObsID", model=model).get_value_ids().get_names()
    names = df.loc[ws_incl, class_col]
    class_names = m_info.classnames
    colors = ['#3973b5', '#6ba552']
    j = 0

    scatter_plots = []
    class_names_lists = []
    toggled_annotations = {}
    if ok=='yes2d':
        scores = np.array(builder.create("tcv", model=model).matrix())
        scores_o = np.array(builder.create("tocv", model=model).matrix())
        scores = np.concatenate((scores, scores_o), axis=0)
        for i in np.unique(classes):
            confidence_ellipse(scores[0, classes == i], scores[1, classes == i], ax, facecolor=colors[j], edgecolor=colors[j], alpha=0.2, linewidth=2)
            sc = ax.scatter(scores[0, classes == i], scores[1, classes == i], color=colors[j], s=45, label=class_names[j], alpha=0.75, edgecolors=colors[j])
            scatter_plots.append(sc)
            class_names_lists.append(names[classes == i])
            j = j + 1

        ax.legend(markerscale=1, fontsize=10)
        ax.grid(visible=None, which='major', axis='both', alpha=0.3)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_xlabel('tCV[1]', fontsize=10)
        ax.set_ylabel('toCV[1]', fontsize=10)

    elif ok=='yes1d':
        scores=np.array(builder.create("tcv", model=model).matrix())
        Num=np.arange(0, scores.shape[1]).reshape(1, scores.shape[1])+1
        scores=np.concatenate(( scores,Num), axis=0)
 
        confidence_level = 0.95
 
        for i in np.unique(classes):
            sc = ax.scatter(scores[0,classes==i],scores[1,classes==i],color=colors[j], s=25,label=class_names[j], alpha=0.75)
            medel=np.mean(scores[0,classes==i])
            stdev=np.std(scores[0,classes==i])#/np.sqrt(scores[1,classes==i].shape[0])
            degrees_freedom = scores[0,classes==i].shape[0] - 1
            confidence_interval = t.interval(confidence_level, degrees_freedom, medel, stdev)
            rect = Rectangle((confidence_interval[0],0),  confidence_interval[1]-confidence_interval[0], Num.shape[1]+1,facecolor=colors[j], edgecolor=colors[j],alpha=.25,linewidth=2)
            ax.add_patch(rect)
            scatter_plots.append(sc)
            class_names_lists.append(names[classes == i])
            j=j+1
 
        ax.legend(markerscale=1, fontsize=10)
        ax.grid(visible=None, which='major', axis='both', alpha=0.3)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_ylabel('Num', fontsize=10)
        ax.set_xlabel('tCV[1]', fontsize=10)
    ax.set_title(m_info.description)
    def create_annotation():
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        return annot

    hover_annotation = create_annotation()
    # persistent_annotations = []
    toggled_annotations = {}

    def update_annot(annot, ind, scatter, class_names_list):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                            " ".join([class_names_list.iloc[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('grey')
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = hover_annotation.get_visible()
        if event.inaxes == ax:
            for scatter, class_names_list in zip(scatter_plots, class_names_lists):
                cont, ind = scatter.contains(event)
                if cont:
                    point_id = (scatter, tuple(ind["ind"]))
                    if point_id in toggled_annotations:
                        hover_annotation.set_visible(False)
                    else:
                        update_annot(hover_annotation, ind, scatter, class_names_list)
                        hover_annotation.set_visible(True)
                        fig.canvas.draw_idle()
                    return
            if vis:
                hover_annotation.set_visible(False)
                fig.canvas.draw_idle()

    def toggle_annotation(event):
        if event.inaxes == ax:
            for scatter, class_names_list in zip(scatter_plots, class_names_lists):
                cont, ind = scatter.contains(event)
                if cont:
                    point_id = (scatter, tuple(ind["ind"]))
                    if point_id in toggled_annotations:
                        toggled_annotations[point_id].set_visible(False)
                        toggled_annotations.pop(point_id)
                    else:
                        if point_id in toggled_annotations:
                            annot = toggled_annotations[point_id]
                        else:
                            annot = create_annotation()
                            toggled_annotations[point_id] = annot
                        update_annot(annot, ind, scatter, class_names_list)
                        annot.set_visible(True)
                        hover_annotation.set_visible(False)
                    fig.canvas.draw_idle()
                    return

    fig.canvas.mpl_connect("button_press_event", toggle_annotation)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    ax.set_aspect('auto')
    fig.tight_layout()

def plot_cv_conf_matrix(bin_models, dataset_name, linked_data, link_option, excl_from_training_set, train_df, class_col, Dnames, excl_test_data, ax, fig):
    project = umetrics.SimcaApp.get_active_project()
    base_workset = project.create_workset(dataset_name)
    base_workset.set_type(umetrics.simca.modeltype.oplsDa)
    excl_indices = list(set(base_workset.get_observations()).difference([i for i in train_df.index.tolist()]).copy())
    excl_indices = [base_workset.get_observations().index(element) for element in excl_indices]
    if excl_indices:
        base_workset.exclude_obs(excl_indices)
    if not excl_test_data:
        base_workset.exclude_obs(excl_from_training_set)
    u_class_names = train_df[class_col].unique().tolist()
    for i, class_name in enumerate(u_class_names):
        c_index = train_df[train_df[class_col] == class_name].index
        base_workset.set_obs_class(c_index, i+1, class_name)
    classes = np.take(base_workset.get_obs_class_phase_numbers(), base_workset.get_included_obs(), axis=0)
    unique_classes = np.unique(classes)
    C=np.array([i for i in range(len(unique_classes))])
    C=np.matlib.repmat(C, len(unique_classes), 1)
    C=C.transpose()
    for i in range(0, len(unique_classes)-1):
        z=linked_data[i,0:2]
        n0=np.where(C[:,i]==z[0])
        n1=np.where(C[:,i]==z[1])
        k=np.max(C,axis=None)+1
        C[n0,i+1]=k
        C[n1,i+1]=k
        if i+2<=np.shape(C)[1]-1:
            C[:,i+2]=C[:,i+1]
    C=np.fliplr(C)
    C2=C*0

    dn1 = dendrogram(linked_data, ax=None, no_plot=True)
    leaves = dn1['leaves']
    # Setup BINARY CLASSIFIERS data
    for i in range(0, len(unique_classes)-1):
        poss=np.unique(np.where(C[:,i]!=C[:,i+1]))
        poss1_before=np.unique(np.where(C[poss,i+1]==np.max(C[poss,i+1])))
        poss2_before=np.unique(np.where(C[poss,i+1]==np.min(C[poss,i+1])))
        if min([leaves.index(poss[p]) for p in poss1_before]) < min([leaves.index(poss[p]) for p in poss2_before]):
            poss1=poss[poss1_before]
            poss2=poss[poss2_before]
        else:
            poss1=poss[poss2_before]
            poss2=poss[poss1_before]
        C2[poss1,i]=1
        C2[poss2,i]=2

    C2=np.delete(C2,-1,axis=1) # MATRIX OF MODELS READ FROM LEFT row corresponds to class columns to models. 1-> class 1, 2 --> class 2 & 0 not in the model.
    m=0
    i=0
    split=0
    decision_limit="YPredcv"
    for m in bin_models:
        i+=1              
        if i == 1:
            pred_set = project.create_predictionset(m)
            pred_set.add_source(umetrics.simca.prediction_source.workset, base_workset)
            pred_set.save('Train_set_pred')
            project.set_predictionset('Train_set_pred')
            pred_ids = [p for p in project.get_predictionset_data().get_obs_names()]
        model_ids = project.data_builder().create("ObsID", model=m).as_dataframe().columns.to_list()
        YPredcv=np.array(project.data_builder().create(decision_limit, model=m).matrix())
        
        if decision_limit=="YPredcv":
            YPredPS=np.array(project.data_builder().create("YPredPS",model=m).matrix())
            # The line below is removed for now but might be considered in the future
        # else:
        #     YPredPS=np.array(project.data_builder().create("Class_ProbabilityPS",model=m).matrix())
        DMODXPS=np.array(project.data_builder().create("DModXPS",model=m).matrix())
        if split==0:
            YHAT=np.zeros((np.size(YPredPS[0,:]),np.size(np.unique(classes))-1))
            DMOD=np.zeros((np.size(DMODXPS[0,:]),np.size(np.unique(classes))-1))
        YHAT[:,split]=YPredPS[0,:]
        DMOD[:, split]=DMODXPS[0,:]
        model_ids_list = list(model_ids)
        pred_ids_list = list(pred_ids)
        matching_indices = [pred_ids_list.index(id) for id in model_ids_list if id in pred_ids_list]
        YHAT[matching_indices, split] = YPredcv[0, :]
        split=split+1
    YHAT2=np.zeros((np.size(YPredPS[0,:]),1))
    DMOD2=np.zeros((np.size(DMODXPS[0,:]),1))
    model_number=np.zeros((np.size(DMODXPS[0,:]),1))
    Ytrue=np.zeros((np.size(YPredPS[0,:]),1))
    for i in range(0,C2.shape[0]):
        class_model=np.array(C2[i,:])
        A=np.array(np.where(class_model==1))
        B=np.array(np.where(class_model==2))
        c=np.where(class_model!=0)
        if A.size*B.size>0:
            Class_i_a=np.where(np.sum(np.sum(YHAT[:,A]>0.5,axis=1),axis=1)==A.shape[1])
            Class_i_b=np.where(np.sum(np.sum(YHAT[:,B]<0.5,axis=1),axis=1)==B.shape[1])
            Class_i=np.intersect1d(Class_i_a,Class_i_b)
            
        elif A.size>0:
            Class_i_a=np.where(np.sum(np.sum(YHAT[:,A]>0.5,axis=1),axis=1)==A.shape[1])
            Class_i=np.intersect1d(Class_i_a,Class_i_a)

        elif B.size>0:
            Class_i_b=np.where(np.sum(np.sum(YHAT[:,B]<0.5,axis=1),axis=1)==B.shape[1])
            Class_i=np.intersect1d(Class_i_b,Class_i_b)
        YHAT2[Class_i,0]=i
        DMOD2[Class_i,0]= np.max(DMOD[np.ix_(Class_i, c[0])], axis=1)
        model_number[Class_i,0] = c[0][np.argmax(DMOD[np.ix_(Class_i, c[0])], axis=1)]+1
    class_index = project.get_predictionset_data('Train_set_pred').get_obs_aliases().index(class_col.split('Metadata_', 1)[1])
    pred_class=project.get_predictionset_data('Train_set_pred').get_obs_names(class_index)
    for i, pred in enumerate(pred_class):
        Ytrue[i,0]=np.squeeze(np.where(np.array(Dnames)==pred))

    yhat_name = [Dnames[int(i)] for i in np.squeeze(YHAT2)]
    # Saves some data to a dataset in SIMCA if someone wants to deep-dive
    data = {"Primary ID": pred_ids,
            "Class CV": pred_class,
            "Predicted CV": yhat_name,
            "Correct prediction CV": ["yes" if c == pc else "no" for c, pc in zip(pred_class, yhat_name)],
            "Max DModXPS CV": np.squeeze(DMOD2),
            "Max DModXPS - Split # CV": np.squeeze(model_number)
            }
    df = pd.DataFrame(data)
    df.set_index("Primary ID", inplace=True)

    imp = umetrics.impdata.ImportData()
    imp.set(df)
    # If it already exists, delete
    try:
        project.delete_dataset('CV-Predictions')
    except:
        pass
    project.create_dataset(imp, str('CV-Predictions'))
    # CALULATES AND SHOWS CONFUSION MATRIX

    # Sort Dnames
    Dnames_sorted = sorted(Dnames)

    # Create a dictionary to map the original indices based on the sorted order
    Dnames_order = {name: i for i, name in enumerate(Dnames)}

    # Initialize confusion matrix
    conf_matrix = np.zeros((len(Dnames_sorted), len(Dnames_sorted)))

    # Build confusion matrix using the alphabetical order
    for i, Dname in enumerate(Dnames_sorted):
        for j, Dname2 in enumerate(Dnames_sorted):
            kt = np.where(Ytrue == Dnames_order[Dname])
            kp = np.where(YHAT2 == Dnames_order[Dname2])
            k = np.intersect1d(kp[0], kt[0])
            conf_matrix[i, j] = k.size

    # Calculate accuracy
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    accuracy_percentage = accuracy * 100
    # and f1_score
    f1 = f1_score(pred_class, yhat_name, average='weighted')

    # Visualization code
    # Create a colormap object based on YlGnBu
    original_cmap = plt.cm.YlGnBu
    # Get the colormap colors and create a new 0 value color (white)
    newcolors = original_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])  # RGBA for white
    newcolors[:1, :] = white  # Set the first row of the colormap to white, i.e.values of 0 => white

    # Create the new colormap with the modified colors
    
    new_cmap = mcolors.ListedColormap(newcolors)
    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

    ax.matshow(normalized_conf_matrix, cmap=new_cmap, alpha=0.5)
    # Populate the confusion matrix
    for i in range(conf_matrix.shape[0]):
        ax.axhline(i-0.5, color='grey', linestyle='-', linewidth=0.5)
        for j in range(conf_matrix.shape[1]):
            ax.axvline(j-0.5, color='grey', linestyle='-', linewidth=0.5)
            # Only populate with non-zero elements
            if conf_matrix[i, j] > 0:

                ax.text(x=j, y=i, s=int(conf_matrix[i, j]), va='center', ha='center', size='xx-large')

    ax.set_xticks(np.arange(len(Dnames_sorted)))
    ax.set_xticklabels(Dnames_sorted, rotation=45, horizontalalignment="left",rotation_mode="anchor")
    ax.set_yticks(np.arange(len(Dnames_sorted)))
    ax.set_yticklabels(Dnames_sorted)
    ax.set_xlabel('Predictions', fontsize=18)
    ax.set_ylabel('Actuals', fontsize=18)
    ax.set_title(f'Confusion Matrix \n CV-predictions \n Accuracy: {accuracy_percentage:.1f}% \n f1-score (weighted): {100*f1:.1f}%', fontsize=18)

    ax.set_aspect('auto')
    fig.tight_layout()

def plot_test_conf_matrix(bin_models, dataset_name, train_dataset_name, linked_data, link_option, excl_from_test_set, train_df, test_df, class_col, Dnames, ax, fig, excl_test_data, excl_from_training_set):
    project = umetrics.SimcaApp.get_active_project()
    base_workset = project.create_workset(train_dataset_name)
    base_workset.set_type(umetrics.simca.modeltype.oplsDa)
    excl_indices = list(set(base_workset.get_observations()).difference([i for i in train_df.index.tolist()]).copy())
    excl_indices = [base_workset.get_observations().index(element) for element in excl_indices]
    if excl_indices:
        base_workset.exclude_obs(excl_indices)
    u_class_names = train_df[class_col].unique().tolist()
    for i, class_name in enumerate(u_class_names):
        c_index = train_df[train_df[class_col] == class_name].index
        base_workset.set_obs_class(c_index, i+1, class_name)
    classes = np.take(base_workset.get_obs_class_phase_numbers(), base_workset.get_included_obs(), axis=0)
    unique_classes = np.unique(classes)
    ## Test data:
    pred_workset = project.create_workset(dataset_name)
    pred_workset.set_type(umetrics.simca.modeltype.oplsDa)
    if dataset_name == train_dataset_name:
        excl_from_test_set = list(set(pred_workset.get_observations()).difference([i for i in test_df.index.tolist()]).copy())
        excl_from_test_set = [pred_workset.get_observations().index(element) for element in excl_from_test_set]
        pred_workset.exclude_obs(excl_from_test_set)
    test_class_names = test_df[class_col].unique().tolist()
    for i, class_name in enumerate(test_class_names):
        c_index = test_df[test_df[class_col] == class_name].index
        pred_workset.set_obs_class(c_index, i+1, class_name)
    test_classes = np.take(pred_workset.get_obs_class_phase_numbers(), pred_workset.get_included_obs(), axis=0)
    test_unique_classes = np.unique(test_classes)
    pred_names = [pred_workset.get_class_phase_names()[ii-1] for ii in test_unique_classes]
    # MATRIX OF MODELS READ FROM LEFT row corresponds to class columns to models. 1-> class 1, 2 --> class 2 & 0 not in the model.
    C=np.array([i for i in range(len(unique_classes))])
    C=np.matlib.repmat(C, len(unique_classes), 1)
    C=C.transpose()
    for i in range(0, len(unique_classes)-1):
        z=linked_data[i,0:2]
        n0=np.where(C[:,i]==z[0])
        n1=np.where(C[:,i]==z[1])
        k=np.max(C,axis=None)+1
        C[n0,i+1]=k
        C[n1,i+1]=k
        if i+2<=np.shape(C)[1]-1:
            C[:,i+2]=C[:,i+1]
    C=np.fliplr(C)
    C2=C*0
    dn1 = dendrogram(linked_data, ax=None, no_plot=True)
    leaves = dn1['leaves']
    # Set up BINARY CLASSIFIERS data
    for i in range(0, len(unique_classes)-1):
        poss=np.unique(np.where(C[:,i]!=C[:,i+1]))
        poss1_before=np.unique(np.where(C[poss,i+1]==np.max(C[poss,i+1])))
        poss2_before=np.unique(np.where(C[poss,i+1]==np.min(C[poss,i+1])))
        if min([leaves.index(poss[p]) for p in poss1_before]) < min([leaves.index(poss[p]) for p in poss2_before]):
            poss1=poss[poss1_before]
            poss2=poss[poss2_before]
        else:
            poss1=poss[poss2_before]
            poss2=poss[poss1_before]

        # poss1=poss[poss1]
        # poss2=poss[poss2]
        C2[poss1,i]=1
        C2[poss2,i]=2

    C2=np.delete(C2,-1,axis=1) # MATRIX OF MODELS READ FROM LEFT row corresponds to class columns to models. 1-> class 1, 2 --> class 2 & 0 not in the model.
    m=0
    i=0
    split=0
    decision_limit="YPredPS"

    m=0
    i=0
    split=0
    # Here prediction sets are created and predicted
    for m in bin_models:
        if i == 0:
            pred_set = project.create_predictionset(m)
            pred_set.add_source(umetrics.simca.prediction_source.dataset, dataset_name, pred_workset.get_included_obs())
            pred_set.save('Test_set_pred')
            project.set_predictionset('Test_set_pred')
            pred_ids = [int(p) for p in project.get_predictionset_data().get_obs_names()]
            i += 1
        YPredPS=np.array(project.data_builder().create(decision_limit, model=m).matrix())
        DMODXPS=np.array(project.data_builder().create("DModXPS",model=m).matrix())
        if split==0:
            YHAT=np.zeros((np.size(YPredPS[0,:]),np.size(np.unique(classes))-1))
            DMOD=np.zeros((np.size(DMODXPS[0,:]),np.size(np.unique(classes))-1))
        YHAT[:,split]=YPredPS[0,:]
        DMOD[:, split]=DMODXPS[0,:]
        split=split+1
    YHAT2=np.zeros((np.size(YPredPS[0,:]),1))
    DMOD2=np.zeros((np.size(DMODXPS[0,:]),1))
    model_number=np.zeros((np.size(DMODXPS[0,:]),1))
    Ytrue=np.zeros((np.size(YPredPS[0,:]),1))
    for i in range(0,C2.shape[0]):
        class_model=np.array(C2[i,:])
        A=np.array(np.where(class_model==1))
        B=np.array(np.where(class_model==2))
        c=np.where(class_model!=0)
        if A.size*B.size>0:
            Class_i_a=np.where(np.sum(np.sum(YHAT[:,A]>0.5,axis=1),axis=1)==A.shape[1])
            Class_i_b=np.where(np.sum(np.sum(YHAT[:,B]<0.5,axis=1),axis=1)==B.shape[1])
            Class_i=np.intersect1d(Class_i_a,Class_i_b)
            
        elif A.size>0:
            Class_i_a=np.where(np.sum(np.sum(YHAT[:,A]>0.5,axis=1),axis=1)==A.shape[1])
            Class_i=np.intersect1d(Class_i_a,Class_i_a)

        elif B.size>0:
            Class_i_b=np.where(np.sum(np.sum(YHAT[:,B]<0.5,axis=1),axis=1)==B.shape[1])
            Class_i=np.intersect1d(Class_i_b,Class_i_b)
        YHAT2[Class_i,0]=i
        DMOD2[Class_i,0]= np.max(DMOD[np.ix_(Class_i, c[0])], axis=1)
        model_number[Class_i,0] = c[0][np.argmax(DMOD[np.ix_(Class_i, c[0])], axis=1)]+1
    class_index = project.get_predictionset_data('Test_set_pred').get_obs_aliases().index(class_col.split('Metadata_', 1)[1])
    pred_class=project.get_predictionset_data('Test_set_pred').get_obs_names(class_index)
    for i, pred in enumerate(pred_class):
        Ytrue[i,0]=np.squeeze(np.where(np.array(pred_names)==pred))

    yhat_name = [Dnames[int(i)] for i in np.squeeze(YHAT2)]
    # Save prediction data to a SIMCA dataset for further analysis
    data = {"Primary ID": pred_ids,
            "Class PS": pred_class,
            "Predicted PS": yhat_name,
            "Correct prediction PS": ["yes" if c == pc else "no" if pc in pred_class else "N/A" for c, pc in zip(pred_class, yhat_name)],
            "Max DModPS PS": np.squeeze(DMOD2),
            "Max DModXPS - Split # PS": np.squeeze(model_number)
            }
    df = pd.DataFrame(data)
    df.set_index("Primary ID", inplace=True)
    imp = umetrics.impdata.ImportData()
    imp.set(df)
    # See if it exists and delete if so
    try:
        project.delete_dataset('Test-Predictions')
    except:
        pass
    project.create_dataset(imp, str('Test-Predictions'))

    # CALULATES AND SHOWS CONFUSION MATRIX
    # Sort names
    pred_names_sorted = sorted(pred_names)
    Dnames_sorted = sorted(Dnames)

    # Create dictionaries for the original index based on the sorted order
    pred_order = {name: i for i, name in enumerate(pred_names)}
    Dnames_order = {name: i for i, name in enumerate(Dnames)}

    # Initialize confusion matrix
    conf_matrix = np.zeros((len(pred_names_sorted), len(Dnames_sorted)))

    # Build confusion matrix using the desired alphabetical order
    for i, pred_name in enumerate(pred_names_sorted):
        for j, Dname in enumerate(Dnames_sorted):
            kt = np.where(Ytrue == pred_order[pred_name])
            kp = np.where(YHAT2 == Dnames_order[Dname])
            k = np.intersect1d(kp[0], kt[0])
            conf_matrix[i, j] = k.size

    # Calculate accuracy
    if set(np.unique(pred_names)) == set(u_class_names):
        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
        accuracy_percentage = accuracy * 100
        f1 = f1_score(pred_class, yhat_name, average='weighted')

    # Create a colormap object based on YlGnBu
    original_cmap = plt.cm.YlGnBu
    # Get the colormap colors and create a new 0 value color (white)
    newcolors = original_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])  # RGBA for white
    newcolors[:1, :] = white  # Set the first row of the colormap to white, i.e. for zero elements => white

    # Create the new colormap with the modified colors
    new_cmap = mcolors.ListedColormap(newcolors)
    normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

    ax.matshow(normalized_conf_matrix, cmap=new_cmap, alpha=0.5)
    # Build confusion matrix plot
    for i in range(conf_matrix.shape[0]):
        ax.axhline(i-0.5, color='grey', linestyle='-', linewidth=0.5)
        for j in range(conf_matrix.shape[1]):
            ax.axvline(j-0.5, color='grey', linestyle='-', linewidth=0.5)
            # Populate confusion matrix with non-zero elements
            if conf_matrix[i, j] > 0:
                ax.text(x=j, y=i, s=int(conf_matrix[i, j]), va='center', ha='center', size='xx-large')

    ax.set_xticks(np.arange(len(Dnames_sorted)))
    ax.set_xticklabels(Dnames_sorted, rotation=45, horizontalalignment="left",rotation_mode="anchor")
    ax.set_yticks(np.arange(len(pred_names_sorted)))
    ax.set_yticklabels(pred_names_sorted)
    ax.set_xlabel('Predictions', fontsize=18)
    ax.set_ylabel('Actuals', fontsize=18)
    # ax.set_title(f'Confusion Matrix \n Test-predictions \n Accuracy: {accuracy_percentage:.1f}% \n f1-score (weighted): {100*f1:.1f}%', fontsize=18)
    if set(np.unique(pred_names)) == set(u_class_names):
        ax.set_title(f'Accuracy: {accuracy_percentage:.1f}% \n f1-score (weighted): {100*f1:.1f}%', fontsize=18)

    ax.set_aspect('auto')
    fig.tight_layout()

def plot_volcano(model, ax, fig, df, class_col):
    ax.cla()
    def neg_log_transform(p, cap=0):
        return -np.log10(p)
    project = umetrics.SimcaApp.get_active_project()
    p_vector = calc_p_vector(project=project, model=model)
    fc_vector = calc_fc_vector(project=project, model=model)

    n_xvar = len(p_vector)
    val_ids = project.data_builder().create("VarDS", model=model).series_names()[0:n_xvar]
    val_ids = [val_id.split(f"M{model}.")[1] for val_id in val_ids]
    df = pd.DataFrame({'Var': val_ids, 'P.Val': p_vector, 'FC':fc_vector})

    marker_size = 50

    p_limit = 0.01
    fc_limit = 1.5

    alpha_val=0.8
    scatter_plots = []
    var_names_lists = []
    down = df[(df['FC']<=1/fc_limit)&(df['P.Val']<=p_limit)]
    if len(down) > 0:
        sc = ax.scatter(x=down['FC'].apply(np.log2),
                    y=down['P.Val'].apply(neg_log_transform),
                    s=marker_size, label="High in Left", color="#3973b5", alpha=alpha_val, edgecolors='grey')
        scatter_plots.append(sc)
        var_names_lists.append(down['Var'].tolist())

    up = df[(df['FC']>=fc_limit)&(df['P.Val']<=p_limit)]
    if len(up) > 0:
        sc = ax.scatter(x=up['FC'].apply(np.log2),
                    y=up['P.Val'].apply(neg_log_transform),
                    s=marker_size, label="High in Right", color="#6ba552", alpha=alpha_val)
        scatter_plots.append(sc)
        var_names_lists.append(up['Var'].tolist())

    not_significant = ~((df['FC'] <= 1/fc_limit) & (df['P.Val'] <= p_limit)) & ~((df['FC'] >= fc_limit) & (df['P.Val'] <= p_limit))
    if any(not_significant):
        sc = ax.scatter(x=df[not_significant]["FC"].apply(np.log2),
                    y=df[not_significant]["P.Val"].apply(neg_log_transform),
                    s=5, label="Not significant")
        scatter_plots.append(sc)
        var_names_lists.append(df[not_significant]['Var'].tolist())

    ax.legend()
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10P.val")
    ax.axvline(np.log2(fc_limit),color="green",linestyle="--", alpha=0.5)
    ax.axvline(-np.log2(fc_limit),color="green",linestyle="--", alpha=0.5)
    ax.axhline(-np.log10(p_limit),color="green",linestyle="--", alpha=0.5)

    def create_annotation():
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        return annot

    hover_annotation = create_annotation()
    toggled_annotations = {}

    def update_annot(annot, ind, scatter, class_names_list):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                            " ".join([class_names_list[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('grey')
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = hover_annotation.get_visible()
        if event.inaxes == ax:
            for scatter, var_names_list in zip(scatter_plots, var_names_lists):
                cont, ind = scatter.contains(event)
                if cont:
                    point_id = (scatter, tuple(ind["ind"]))
                    if point_id in toggled_annotations:
                        hover_annotation.set_visible(False)
                    else:
                        update_annot(hover_annotation, ind, scatter, var_names_list)
                        hover_annotation.set_visible(True)
                        fig.canvas.draw_idle()
                    return
            if vis:
                hover_annotation.set_visible(False)
                fig.canvas.draw_idle()

    def toggle_annotation(event):
        if event.inaxes == ax:
            for scatter, var_names_list in zip(scatter_plots, var_names_lists):
                cont, ind = scatter.contains(event)
                if cont:
                    point_id = (scatter, tuple(ind["ind"]))
                    if point_id in toggled_annotations:
                        toggled_annotations[point_id].set_visible(False)
                        toggled_annotations.pop(point_id)
                    else:
                        if point_id in toggled_annotations:
                            annot = toggled_annotations[point_id]
                        else:
                            annot = create_annotation()
                            toggled_annotations[point_id] = annot
                        update_annot(annot, ind, scatter, var_names_list)
                        annot.set_visible(True)
                        hover_annotation.set_visible(False)
                    fig.canvas.draw_idle()
                    return

    fig.canvas.mpl_connect("button_press_event", toggle_annotation)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    ax.set_aspect('auto')
    fig.tight_layout()

def plot_volcano_log2(model, ax, fig, df, class_col):
    ax.cla()
    def neg_log_transform(p, cap=0):
        return -np.log10(p)
    project = umetrics.SimcaApp.get_active_project()
    p_vector = calc_p_vector(project=project, model=model)
    fc_vector = calc_fc_log2_vector(project=project, model=model)

    n_xvar = len(p_vector)
    val_ids = project.data_builder().create("VarDS", model=model).series_names()[0:n_xvar]
    val_ids = [val_id.split(f"M{model}.")[1] for val_id in val_ids]
    df = pd.DataFrame({'Var': val_ids, 'P.Val': p_vector, 'FC':fc_vector})

    marker_size = 50

    p_limit = 0.01
    fc_limit = 1.5

    alpha_val=0.8
    scatter_plots = []
    var_names_lists = []
    down = df[(df['FC']<=1/fc_limit)&(df['P.Val']<=p_limit)]
    if len(down) > 0:
        sc = ax.scatter(x=down['FC'].apply(np.log2),
                    y=down['P.Val'].apply(neg_log_transform),
                    s=marker_size, label="High in Left", color="#3973b5", alpha=alpha_val, edgecolors='grey')
        scatter_plots.append(sc)
        var_names_lists.append(down['Var'].tolist())

    up = df[(df['FC']>=fc_limit)&(df['P.Val']<=p_limit)]
    if len(up) > 0:
        sc = ax.scatter(x=up['FC'].apply(np.log2),
                    y=up['P.Val'].apply(neg_log_transform),
                    s=marker_size, label="High in Right", color="#6ba552", alpha=alpha_val)
        scatter_plots.append(sc)
        var_names_lists.append(up['Var'].tolist())

    not_significant = ~((df['FC'] <= 1/fc_limit) & (df['P.Val'] <= p_limit)) & ~((df['FC'] >= fc_limit) & (df['P.Val'] <= p_limit))
    if any(not_significant):
        sc = ax.scatter(x=df[not_significant]["FC"].apply(np.log2),
                    y=df[not_significant]["P.Val"].apply(neg_log_transform),
                    s=5, label="Not significant")
        scatter_plots.append(sc)
        var_names_lists.append(df[not_significant]['Var'].tolist())

    ax.legend()
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10P.val")
    ax.axvline(np.log2(fc_limit),color="green",linestyle="--", alpha=0.5)
    ax.axvline(-np.log2(fc_limit),color="green",linestyle="--", alpha=0.5)
    ax.axhline(-np.log10(p_limit),color="green",linestyle="--", alpha=0.5)

    def create_annotation():
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        return annot

    hover_annotation = create_annotation()
    toggled_annotations = {}

    def update_annot(annot, ind, scatter, class_names_list):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                            " ".join([class_names_list[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('grey')
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = hover_annotation.get_visible()
        if event.inaxes == ax:
            for scatter, var_names_list in zip(scatter_plots, var_names_lists):
                cont, ind = scatter.contains(event)
                if cont:
                    point_id = (scatter, tuple(ind["ind"]))
                    if point_id in toggled_annotations:
                        hover_annotation.set_visible(False)
                    else:
                        update_annot(hover_annotation, ind, scatter, var_names_list)
                        hover_annotation.set_visible(True)
                        fig.canvas.draw_idle()
                    return
            if vis:
                hover_annotation.set_visible(False)
                fig.canvas.draw_idle()

    def toggle_annotation(event):
        if event.inaxes == ax:
            for scatter, var_names_list in zip(scatter_plots, var_names_lists):
                cont, ind = scatter.contains(event)
                if cont:
                    point_id = (scatter, tuple(ind["ind"]))
                    if point_id in toggled_annotations:
                        toggled_annotations[point_id].set_visible(False)
                        toggled_annotations.pop(point_id)
                    else:
                        if point_id in toggled_annotations:
                            annot = toggled_annotations[point_id]
                        else:
                            annot = create_annotation()
                            toggled_annotations[point_id] = annot
                        update_annot(annot, ind, scatter, var_names_list)
                        annot.set_visible(True)
                        hover_annotation.set_visible(False)
                    fig.canvas.draw_idle()
                    return

    fig.canvas.mpl_connect("button_press_event", toggle_annotation)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    ax.set_aspect('auto')
    fig.tight_layout()

def calc_fc_log2_vector(project, model):
    # Calculates the fold-change vector (used for volcano plot)
    ## Note: This doesn't work properly if variables have  negative values
    Y=np.array(project.data_builder().create("YVar",model=model).matrix())
    X=np.array(project.data_builder().create("XVar",model=model).matrix())
    C=np.array(project.data_builder().create("c",model=model).matrix())
    FC = []
    for i in range(0,X.shape[0]):
        x=X[i,:]
        m0=np.mean(pow(2,x[np.where((Y[0,:])==np.min(Y[0,:]))]))
        m1=np.mean(pow(2,x[np.where((Y[0,:])==np.max(Y[0,:]))]))
        # WLp[0][i]=p
        # if  m1/m0>0:
        if min([m1, m0])>0:
            if C[0,0]>0:
                FC.append(m1/m0)
            else:
                FC.append(m0/m1)
        else:
            FC.append(1)
    return FC

def calc_fc_vector(project, model):
    # Calculates the fold-change vector (used for volcano plot)
    ## Note: This doesn't work properly if variables have  negative values
    Y=np.array(project.data_builder().create("YVar",model=model).matrix())
    X=np.array(project.data_builder().create("XVar",model=model).matrix())
    C=np.array(project.data_builder().create("c",model=model).matrix())
    FC = []
    for i in range(0,X.shape[0]):
        x=X[i,:]
        m0=np.mean(x[np.where((Y[0,:])==np.min(Y[0,:]))])
        m1=np.mean(x[np.where((Y[0,:])==np.max(Y[0,:]))])
        # WLp[0][i]=p
        # if  m1/m0>0:
        if min([m1, m0])>0:
            if C[0,0]>0:
                FC.append(m1/m0)
            else:
                FC.append(m0/m1)
        else:
            FC.append(1)
    return FC

def calc_p_vector(project, model):
    # Calculate p-values (used for volcano plot)
    Y=np.array(project.data_builder().create("YVar",model=model).matrix())
    if project.get_model_info(model).orthogonalinX>0:
        to=np.array(project.data_builder().create("to",model=model).matrix())
    elif project.get_model_info(model).orthogonalinX==0:
        to=np.array([])
    X=np.array(project.data_builder().create("XVar",model=model).matrix())
    y=np.array(Y[0,:])
    to=np.squeeze(to)
    if max(to.shape)>0:
        if len(np.unique(y))>1:
            Z=np.column_stack((to.transpose(),np.ones(len(y)).transpose(),y))
        else:
            Z=np.column_stack((to.transpose(),y))
    else:
        if len(np.unique(y))>1:
            Z=np.column_stack((np.ones(len(y)).transpose(),y))
        else:
            Z=np.reshape(y,(y.shape[0],1))
    WLp = []
    for i in range(0,X.shape[0]):
        yc=X[i,:]

        non_nan_indices = np.where(~np.isnan(yc))[0]
        yc = yc[non_nan_indices]
        z = Z[non_nan_indices, :]

        B=np.dot(z.T.dot(yc),np.linalg.inv(np.matmul(z.T, z)))
        yhat=z.dot(B)
        ydiff=yc-yhat
        rsd=np.sqrt(np.sum(ydiff**2)/(z.shape[0]-z.shape[1]))
        se=(1/np.sqrt(np.diag(np.matmul(z.T, z)))).dot(rsd)
        T=B[-1]/se[-1]
        p=2*t.cdf(-abs(T),z.shape[0]-z.shape[1])
        if p<1.175494351e-38:
            p=1.175494351e-38
        WLp.append(p)
    return WLp

def check_bin_models(bin_models):
    # Check if we have binary models in each split and if this are the same as the variable "bin_models"
    project = umetrics.SimcaApp.get_active_project()
    check_list = []
    bin_simca_models = [sm.number for sm in project.get_model_infos() if sm.description[0:6] == "Binary"]
    if len(bin_simca_models) == len(bin_models):
        for simca_model, bin_model in zip(bin_simca_models, bin_models):
            check_list.append(simca_model == bin_model)
        if check_list:
            return all(check_list)
        else:
            return False
    else:
        return False

def check_ovo_models(bin_models):
    # Check if we have one-vs-one models for each class combination and if this are the same as the variable "bin_models"
    project = umetrics.SimcaApp.get_active_project()
    check_list = []
    bin_simca_models = [sm.number for sm in project.get_model_infos() if sm.description[0:6]=="One vs"]
    if len(bin_simca_models) == len(bin_models):
        for simca_model, bin_model in zip(bin_simca_models, bin_models):
            check_list.append(simca_model == bin_model)
        return all(check_list)
    else:
        return False

def clear_ovo_models():
    # Remove all one-vs-one models
    project = umetrics.SimcaApp.get_active_project()
    for model in project.get_model_infos():
        if model.description[0:3]=="One": 
            project.delete_model(model.number)

def clear_bin_models():
    # Remove all binary models
    project = umetrics.SimcaApp.get_active_project()
    for model in project.get_model_infos():
        if model.description[0:6]=="Binary": 
            project.delete_model(model.number)

## Set reproducibility for when train and test sets are created
random.seed(42)
class SequentialTabsGUI(tk.Tk):
    def __init__(self, dfs=None):
        super().__init__()
        # Configure the main window
        self.title("OPLS-HDA Wizard")
        self.geometry("1200x800")
        # Create a figure for the dendrogram
        self.heat_fig, self.heat_ax = plt.subplots()
        self.fig, self.ax = plt.subplots()
        self.score_fig, self.score_ax = plt.subplots()
        self.hs_fig, self.hs_ax = plt.subplots()
        # self.score_fig, self.score_ax = plt.subplots(figsize=(5, 4))
        # self.dendrogram_canvas = None
        self.dfs = [*dfs.values()]
        self.df_names = [*dfs.keys()]
        self.df = None
        self.df_index = None
        self.distance_matrix = None
        self.column_display_map = {}
        self.how_fit = None
        self.how_scale = None
        self.bin_models = []
        self.ovo_models = []
        self.sel_df = None
        self.n_components = None
        #
        self.incl_class_dict = None
        self.incl_classes = None
        self.excl_classes = None
        self.class_col = None
        self.sel_column = None
        self.train_df = None
        self.test_df = None
        self.sel_test_set_option = None
        self.skip_option = None
        self.test_df_option = None
        self.sel_split_option = None
        self.Dnames = None
        self.link_choice = None

        # Define the number of steps
        self.num_steps = 4
        
        # Initialize completion states for each step
        self.completion_states = [tk.BooleanVar(value=False) for _ in range(self.num_steps - 1)]

        # Create a Notebook widget
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)


        # Initialize tab frames and add to the notebook
        self.tab_frames = [ttk.Frame(self.notebook) for _ in range(self.num_steps)]
        step_names = [f"Step {i + 1}" for i in range(self.num_steps)]
        for idx, frame in enumerate(self.tab_frames):
            self.notebook.add(frame, text=step_names[idx])

        # Initialize list of dicts for all steps to track changes and store data
        self.entries = []
        self.original_values = []
        for i in range(self.num_steps):
            self.entries.append({})
            self.original_values.append({})

        # Fetch existing data from SIMCA storage...
        project = umetrics.SimcaApp.get_active_project()
        storage = project.get_project_storage()
        self.storage = storage
        self.entries = self.storage.get('opls_hda_entries', [])
        self.original_values = []
        # ... if it exists
        if not self.entries:
            for i in range(self.num_steps):
                self.entries.append({})
                self.original_values.append({})
            self.entries[0].update({'tab_state': 'normal'})
            # Initially disable all tabs except Step 1
            for i in range(1, self.num_steps):
                self.notebook.tab(i, state='disabled')
                self.entries[i].update({'tab_state': 'disabled'})
            self.notebook.select(0)
        else:
            for i in range(self.num_steps):
                self.original_values.append({})
                self.notebook.tab(i, state=self.entries[i]['tab_state'])
            self.df_index = self.entries[0].get('dataframe', None)
            if self.df_index is not None:
                self.df = self.dfs[self.df_index]
            self.sel_df = self.entries[0].get('selected_df', None)
            self.incl_class_dict = self.entries[0].get('incl_class_dict', None)
            self.incl_classes = self.entries[0].get('incl_classes', None)
            self.excl_classes = self.entries[0].get('excl_classes', None)
            self.class_col = self.entries[0].get('class_col', None)
            self.sel_column = self.entries[0].get('selected_col', None)
            self.notebook.select(self.entries[0]['prev_idx'])
            self.train_df = self.entries[1].get('train_df', None)
            self.test_df = self.entries[1].get('test_df', None)
            self.sel_test_set_option = self.entries[1].get('test_set_option', None)
            self.skip_option = self.entries[1].get('skip_option', None)
            self.test_df_option = self.entries[1].get('test_df_option', None)
            self.sel_split_option = self.entries[1].get('sel_split_option', None)
            self.n_components = self.entries[2].get('n_components', None)
            self.how_fit = self.entries[2].get('how_fit', None)
            self.how_scale = self.entries[2].get('how_scale', None)
            self.distance_matrix = self.entries[2].get('distance_matrix', None)
            self.Dnames = self.entries[2].get('Dnames', None)
            self.ovo_models = self.entries[2].get('ovo_models', None)
            self.link_choice = self.entries[3].get('linkage_choice', None)
            self.bin_models = self.entries[3].get('bin_models', [])

        self.previous_tab_index = int(self.notebook.index("current"))
        self.create_steps_content()
        # Handle the window close event to ensure proper exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Resize window using the key 'r'
        def keydown(e):
            if e.char=='r':
                if self.wm_state()=='zoomed':
                    self.wm_state('normal')
                else:
                    self.wm_state('zoomed')
        self.bind("<KeyPress>", keydown)

    # Create all the contents
    def create_steps_content(self):
        def setup_step_1(frame):
            self.create_df_selection(frame)
            self.create_column_selection(frame)
            self.add_class_selection_dual_listbox(frame)

        def setup_step_2(frame):
            self.create_test_set_options(frame)
            # self.add_complete_button(frame, 2)

        def setup_step_3(frame):
            self.choose_fit_method(frame)
            # self.add_continue_button(self.ovo_label_frame, 3)

        def setup_step_4(frame):
            self.create_linkage_selection(frame)

        step_setups = [func for name, func in sorted(locals().items()) if name.startswith("setup_step_")]
        for i, setup in enumerate(step_setups):
            frame = self.tab_frames[i]
            setup(frame)

    def create_df_selection(self, frame):
        label = ttk.Label(frame, text="Select the model dataset:")
        label.pack(side="top", pady=10, padx=10, anchor='nw')
        self.selected_df = tk.StringVar()
        name_length = 0
        for names in self.df_names:
            if len(names) > name_length:
                name_length = len(names)
        self.df_combobox = ttk.Combobox(frame, textvariable=self.selected_df, values=self.df_names, state="readonly", width=name_length)
        self.df_combobox.pack(side="top", pady=10,padx=10, anchor='nw')
        self.df_combobox.bind("<<ComboboxSelected>>", self.on_df_selected)
        if self.df_index is not None:
            self.selected_df.set(self.df_names[self.df_index])
            self.df_combobox.set(self.selected_df.get())

    def train_data_selection(self):
        self.incl_classes = [self.incl_class_dict[self.include_listbox.get(i)] for i in range(self.include_listbox.size())]
        self.excl_classes = [self.incl_class_dict[self.exclude_listbox.get(i)] for i in range(self.exclude_listbox.size())]
        self.sel_df = self.df.query(f'`{self.class_col}` == @self.incl_classes').copy()
        self.entries[0].update({'selected_df': self.sel_df})
        self.entries[0].update({'dataframe' : self.df_index})
        self.entries[0].update({'class_col' : self.class_col})
        self.entries[0].update({'selected_col' : self.selected_column.get()})
        self.entries[0].update({'incl_classes' : [self.include_listbox.get(i) for i in range(self.include_listbox.size())]})
        self.entries[0].update({'excl_classes' : [self.exclude_listbox.get(i) for i in range(self.exclude_listbox.size())]})
        self.look_for_change(int(self.previous_tab_index))
        self.complete_button_one.config(state=tk.NORMAL)

    def on_df_selected(self, event):
        self.train_confirm_button.config(state=tk.DISABLED)
        self.complete_button_one.config(state=tk.DISABLED)
        selected_idx = self.df_combobox.current()
        self.df = self.dfs[selected_idx]
        self.df_index = selected_idx
        self.update_column_selection()
        self.exclude_listbox.delete(0, tk.END)
        self.include_listbox.delete(0, tk.END)

    def create_column_selection(self, frame):
        self.column_label = ttk.Label(frame, text="Select the class column:")
        self.column_label.pack(side="top",pady=10, padx=10, anchor='nw')
        self.selected_column = tk.StringVar()
        if self.sel_column is not None:
            self.selected_column.set(self.sel_column)
        self.column_combobox = ttk.Combobox(frame, textvariable=self.selected_column, state="readonly")
        self.column_combobox.pack(side="top",pady=10, padx=10, anchor='nw')
        self.column_combobox.bind("<<ComboboxSelected>>", self.on_column_selected)
    
    def on_column_selected(self, event):
        self.train_confirm_button.config(state=tk.NORMAL)
        self.complete_button_one.config(state=tk.NORMAL)
        self.class_col = self.column_display_map.get(self.selected_column.get())
        self.sel_column = self.selected_column.get()
        self.update_class_lists()
        
    def update_column_selection(self):
        column_names = list(self.df.columns)
        n_unique_classes = [len(self.df[col].unique()) for col in column_names if col.startswith('Metadata')]
        col_names_and_n = [f"{col} ({n})" for col, n in zip(column_names, n_unique_classes)]
        self.column_display_map = {display_name: col for display_name, col in zip(col_names_and_n, column_names)}
        name_length = 0
        for names in col_names_and_n:
            if len(names) > name_length:
                name_length = len(names)
        self.column_combobox.config(values=col_names_and_n, width=name_length)
        self.column_combobox.set("")
        self.update_class_lists()

    def add_class_selection_dual_listbox(self, frame):
        self.class_selection_frame = ttk.Frame(frame)
        self.class_selection_frame.pack(side="top",pady=1, fill='x', expand=True, anchor='nw')

        # Labels for the listboxes
        include_label = ttk.Label(self.class_selection_frame, text="Include")
        include_label.grid(row=0, column=2, padx=10, pady=1)

        exclude_label = ttk.Label(self.class_selection_frame, text="Exclude")
        exclude_label.grid(row=0, column=0, padx=10, pady=1)

        # Scrollbars for listboxes
        self.include_scrollbar = tk.Scrollbar(self.class_selection_frame, orient=tk.VERTICAL)
        self.exclude_scrollbar = tk.Scrollbar(self.class_selection_frame, orient=tk.VERTICAL)
        self.include_scrollbar_horizont = tk.Scrollbar(self.class_selection_frame, orient=tk.HORIZONTAL)
        self.exclude_scrollbar_horizont = tk.Scrollbar(self.class_selection_frame, orient=tk.HORIZONTAL)

        # Include listbox
        self.include_listbox = tk.Listbox(self.class_selection_frame, selectmode='extended')
        self.include_listbox.grid(row=1, column=2, padx=10, pady=1, sticky='n')
        self.include_listbox.configure(yscrollcommand=self.include_scrollbar.set, xscrollcommand=self.include_scrollbar_horizont.set)
        self.include_scrollbar.config(command=self.include_listbox.yview)
        self.include_scrollbar.grid(row=1, column=3, sticky='nsw', pady=1)
        self.include_scrollbar_horizont.config(command=self.include_listbox.xview)
        self.include_scrollbar_horizont.grid(row=2, column=2, sticky='new', padx=10)

        # Exclude listbox
        self.exclude_listbox = tk.Listbox(self.class_selection_frame, selectmode='extended')
        self.exclude_listbox.grid(row=1, column=0, padx=(10, 0), pady=1, sticky="e")
        self.exclude_listbox.configure(yscrollcommand=self.exclude_scrollbar.set, xscrollcommand=self.exclude_scrollbar_horizont.set)
        self.exclude_scrollbar.config(command=self.exclude_listbox.yview)
        self.exclude_scrollbar.grid(row=1, column=1, sticky='nsw', pady=1, padx=(0, 5))
        self.exclude_scrollbar_horizont.config(command=self.exclude_listbox.xview)
        self.exclude_scrollbar_horizont.grid(row=2, column=0, sticky='new', padx=(10, 5))

        if self.incl_classes:
            for cls in self.incl_classes:
                self.include_listbox.insert(tk.END, f"{cls}")
                self.entries[0].update({'incl_classes' : [self.include_listbox.get(i) for i in range(self.include_listbox.size())]})

        if self.excl_classes:
            for cls in self.excl_classes:
                self.exclude_listbox.insert(tk.END, f"{cls}")
                self.entries[0].update({'excl_classes' : [self.exclude_listbox.get(i) for i in range(self.exclude_listbox.size())]})

        # Buttons for moving items between listboxes
        add_button = ttk.Button(self.class_selection_frame, text="Add >", command=self.add_to_include)
        add_button.grid(row=0, column=1, padx=10, pady=10, sticky='n')

        remove_button = ttk.Button(self.class_selection_frame, text="< Remove", command=self.remove_from_include)
        remove_button.grid(row=2, column=1, padx=10, pady=10, sticky='n')
        self.train_confirm_button = ttk.Button(frame, text="Confirm", command=lambda : self.train_data_selection())
        self.train_confirm_button.pack(anchor='nw', padx=10)

        step_index = 1
        self.complete_button_one = ttk.Button(frame, text=f"Complete Step {step_index}",
                                     command=lambda idx=step_index-1: self.complete_step(step_index-1))
        self.complete_button_one.pack(side='left', padx=10, pady=10, anchor='nw')
        self.complete_button_one.config(state=tk.DISABLED)
        if self.incl_classes:
            self.train_confirm_button.config(state=tk.NORMAL)
            self.complete_button_one.config(state=tk.NORMAL)
        else:
            self.train_confirm_button.config(state=tk.DISABLED)
            self.complete_button_one.config(state=tk.DISABLED)

    def update_class_lists(self):
        selected_column = self.class_col
        self.incl_class_dict = {}
        if selected_column:
            unique_classes = self.df[selected_column].unique()
            self.exclude_listbox.delete(0, tk.END)
            self.include_listbox.delete(0, tk.END)
            for cls in unique_classes:
                n_cls = len(self.df[self.df[selected_column] == cls])
                self.include_listbox.insert(tk.END, f"{cls} ({n_cls})")
                self.incl_class_dict.update({f"{cls} ({n_cls})": cls})
            self.entries[0].update({'incl_class_dict': self.incl_class_dict})

    def add_to_include(self):
        selected_indices = self.exclude_listbox.curselection()
        for index in selected_indices[::-1]:
            value = self.exclude_listbox.get(index)
            self.include_listbox.insert(tk.END, value)
            self.exclude_listbox.delete(index)

    def remove_from_include(self):
        selected_indices = self.include_listbox.curselection()
        for index in selected_indices[::-1]:
            value = self.include_listbox.get(index)
            self.exclude_listbox.insert(tk.END, value)
            self.include_listbox.delete(index)

    def create_test_set_options(self, frame):
        label = ttk.Label(frame, text="Choose an option for the test set:")
        label.pack(pady=10, padx=10, anchor='nw')

        self.test_set_option = tk.StringVar()
        
        options = [("Create test set from the chosen training set", "create"),
                ("Choose another dataframe as test set", "choose"),
                ("Skip test set", "skip")]

        for text, value in options:
            radio_button = ttk.Radiobutton(frame, text=text, variable=self.test_set_option, value=value, 
                                        command=lambda value=value: self.on_test_set_option_change(value))
            radio_button.pack(anchor="nw", padx=10)
        # Sub-frame for dynamic content
        self.dynamic_frame = ttk.Frame(frame)
        self.dynamic_frame.pack(fill="both", expand=True)
        step_index = 2
        self.complete_button_two = ttk.Button(frame, text=f"Complete Step {step_index}",
                                     command=lambda idx=step_index-1: self.complete_step(step_index-1))
        self.complete_button_two.pack(side='left', padx=10, pady=10, anchor='nw')
        self.complete_button_two.config(state=tk.DISABLED)

        if self.sel_test_set_option is not None:
            self.test_set_option.set(self.sel_test_set_option)
            self.on_test_set_option_change(self.sel_test_set_option)
            self.complete_button_two.config(state=tk.NORMAL)

    def on_test_set_option_change(self, selected_option):
    # Clear only the dynamic frame
        self.clear_frame(self.dynamic_frame)
        if selected_option == "create":
            label = ttk.Label(self.dynamic_frame, text="Choose how to split the data in train/test sets:")
            label.pack(pady=10, padx=10, anchor='nw')
            self.split_option = tk.StringVar()
            # for text, value in [('Stratified class split', 'strat_class'), ('Stratified based on other metadata', 'strat_meta')]:
            for text, value in [('Stratified class split', 'strat_class')]:
                radio_button = ttk.Radiobutton(self.dynamic_frame, text=text, variable=self.split_option, value=value, 
                                            command=lambda value=value: self.strat_split_from_training_data(value))
                radio_button.pack(anchor="w", padx=10)
            if self.sel_split_option is not None:
                self.split_option.set(self.sel_split_option)
            self.entries[1].update({'sel_split_option': self.split_option.get()})
            self.entries[1].update({'test_df_option': False})
        elif selected_option == "choose":
            # Additional frame manipulations
            label = ttk.Label(self.dynamic_frame, text="Select test DataFrame:")
            label.pack(pady=10, padx=10, anchor='nw')
            self.selected_test_df = tk.StringVar()
            if self.test_df_option is not None:
                self.selected_test_df.set(self.df_names[self.test_df_option-1])
            name_length = 0
            for names in self.df_names:
                if len(names) > name_length:
                    name_length = len(names)
            self.test_df_combobox = ttk.Combobox(self.dynamic_frame, textvariable=self.selected_test_df, values=self.df_names, state="readonly", width=name_length)
            self.test_df_combobox.pack(pady=10,padx=10, anchor='nw')
            confirm_button = ttk.Button(self.dynamic_frame, text="Confirm", command=lambda : self.select_test_df())
            confirm_button.pack(anchor='nw', padx=10)
            self.entries[1].update({'test_df_option': True})
        elif selected_option == "skip":
            self.entries[1].update({'skip_option': "skip"})
            self.entries[1].update({'test_df_option': True}) ## To skip excl index in model calc later
            confirm_button = ttk.Button(self.dynamic_frame, text="Confirm", command=lambda : self.skip_test_df())
            confirm_button.pack(anchor='nw', padx=10)

        self.dynamic_frame2 = ttk.Frame(self.dynamic_frame)
        self.dynamic_frame2.pack(fill="both", expand=True)
        self.clear_frame(self.dynamic_frame2)
        self.entries[1].update({'test_set_option': selected_option})

    def skip_test_df(self):
        self.clear_frame(self.dynamic_frame2)
        self.test_df = pd.DataFrame()
        self.train_df = self.sel_df.copy()
        self.entries[1].update({'train_df': self.train_df})
        self.entries[1].update({'test_df': pd.DataFrame()})
        self.entries[1].update({'test_df_idx': None})
        self.look_for_change(int(self.previous_tab_index))
        self.complete_button_two.config(state=tk.NORMAL)

    def select_test_df(self):
        self.clear_frame(self.dynamic_frame2)
        selected_idx = self.test_df_combobox.current()
        self.test_df = self.dfs[selected_idx]
        self.train_df = self.sel_df.copy()
        self.entries[1].update({'train_df': self.train_df})
        self.entries[1].update({'test_df': self.test_df})
        self.entries[1].update({'test_df_idx': selected_idx})
        self.look_for_change(int(self.previous_tab_index))
        self.complete_button_two.config(state=tk.NORMAL)
        

    def strat_split_from_training_data(self, strat_split, init=False):
        # Clear only the dynamic frame
        self.clear_frame(self.dynamic_frame2)
        if strat_split == 'strat_meta':
            column_names = list(self.df.columns)
            n_unique_classes = [len(self.df[col].unique()) for col in column_names if col.startswith('Metadata')]
            col_names_and_n = [f"{col} ({n})" for col, n in zip(column_names, n_unique_classes)]
            self.column_display_map = {display_name: col for display_name, col in zip(col_names_and_n, column_names)}
            column_label = ttk.Label(self.dynamic_frame2, text="Select the Metadata column to split:")
            column_label.pack(pady=10, padx=10, anchor='nw')
            self.selected_metadata_col = tk.StringVar()
            column_combobox = ttk.Combobox(self.dynamic_frame2, textvariable=self.selected_metadata_col, state="readonly")
            column_combobox.pack(pady=10, padx=10, anchor='nw')
            name_length = 0
            for names in col_names_and_n:
                if len(names) > name_length:
                    name_length = len(names)
            column_combobox.config(values=col_names_and_n, width=name_length)
            column_combobox.set("")
            self.entries[1].update({'selected_metadata_col': self.selected_metadata_col.get()})

        label = ttk.Label(self.dynamic_frame2, text="Select train and test set size:")
        label.pack(anchor='w', padx=10)
       
        self.train_size = tk.DoubleVar()        
        self.current_val_label = ttk.Label(self.dynamic_frame2, text=f"Train Size: {self.train_size.get():.2f}")
        self.current_val_label.pack(anchor='nw', padx=10)
        self.slider = ttk.Scale(self.dynamic_frame2, from_=0, to=1, orient='horizontal', variable=self.train_size, command=self.slider_changed)
        self.slider.set(0.7)  # Set the default value
        self.slider.pack(anchor="nw", padx=10)
        confirm_button = ttk.Button(self.dynamic_frame2, text="Confirm", command=lambda : self.split_data(strat_split))
        confirm_button.pack(anchor='nw', padx=10)
        self.dynamic_frame3 = ttk.Frame(self.dynamic_frame2)
        self.dynamic_frame3.pack(fill="both", expand=True)

        self.entries[1].update({'split_or_df_option': strat_split})
        self.entries[1].update({'slider_value': self.train_size.get()})
        self.complete_button_two.config(state=tk.NORMAL)

    def split_data(self, strat_split):
        self.clear_frame(self.dynamic_frame3)
        if strat_split == 'strat_class':
            self.train_df, self.test_df = train_test_split(self.sel_df, test_size=round(1-self.train_size.get
            (), 2), train_size=round(self.train_size.get(), 2), 
                                                           stratify=self.sel_df[self.class_col], random_state=42)
            self.entries[1].update({'strat_split': strat_split})
            self.entries[1].update({'train_df': self.train_df})
            self.entries[1].update({'test_df': self.test_df})
            self.entries[1].update({'test_df_idx': self.df_index})
            self.look_for_change(int(self.previous_tab_index))
        elif strat_split == 'strat_meta':
            self.train_df, self.test_df = self.stratified_group_split(self.column_display_map.get(self.selected_metadata_col.get()), test_size=round(1-self.train_size.get(), 2))
            self.entries[1].update({'strat_split': strat_split})
            self.entries[1].update({'train_df': self.train_df})
            self.entries[1].update({'test_df': self.test_df})
            self.entries[1].update({'test_df_idx': self.df_index})
            self.look_for_change(int(self.previous_tab_index))
        
    
    def stratified_group_split(self, group_col, test_size=0.3):
        # Get unique class labels
        class_col = self.class_col
        classes = self.df[class_col].unique()
        
        train_dfs = []
        test_dfs = []

        for class_label in classes:
            # Filter the dataframe for the current class label
            class_df = self.df[self.df[class_col] == class_label]
            
            # Group the data by the metadata column (e.g., breed)
            groups = class_df.groupby(group_col)
            
            # Shuffle the unique groups
            unique_groups = shuffle(list(groups.groups.keys()), random_state=42)
            
            # Calculate the split index
            split_idx = int(len(unique_groups) * (1 - test_size))
            
            # Create training and test sets for the current class label
            train_groups = unique_groups[:split_idx]
            test_groups = unique_groups[split_idx:]
            
            train_df = pd.concat([groups.get_group(g) for g in train_groups])
            test_df = pd.concat([groups.get_group(g) for g in test_groups])
            
            train_dfs.append(train_df)
            test_dfs.append(test_df)
        
        # Concatenate all class-specific splits to get the final training and test sets
        final_train_df = pd.concat(train_dfs)
        final_test_df = pd.concat(test_dfs)
        
        return final_train_df, final_test_df

    def slider_changed(self, event=None):
        self.current_val_label.config(text=f"Train Size: {self.train_size.get():.2f} \nTest Size: {1-self.train_size.get():.2f}")
        self.entries[1].update({'train_size' : round(self.train_size.get(), 2)})
        self.entries[1].update({'test_size' : 1-round(self.train_size.get(), 2)})

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def choose_fit_method(self, frame):
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        self.heat_label_frame = tk.Frame(frame)
        self.heat_label_frame.grid(row=0, column=0, sticky='nsew')

        self.choose_fit_frame = tk.Frame(self.heat_label_frame)
        self.choose_fit_frame.pack(side='left', anchor='nw')

        # Sub-frame for dynamic content
        self.fit_dynamic_frame = ttk.Frame(self.heat_label_frame)
        self.fit_dynamic_frame.pack(side='left', fill="both", expand=True)

        self.choose_scale_frame = tk.Frame(self.heat_label_frame)
        self.choose_scale_frame.pack(side='left', anchor='nw')

        self.ovo_label_frame = tk.Frame(frame)
        self.ovo_label_frame.grid(row=0, column=1, sticky='nsew')
        self.model = None
        self.ovo_score_plot_button = ttk.Button(self.ovo_label_frame, text="CV scores", command=lambda: self.plot_ovo_scores())
        self.ovo_score_plot_button.pack(side='left', anchor='sw', padx=10, pady=10)
        self.ovo_score_plot_button.config(state=tk.DISABLED)
        
        self.ovo_volcano_plot_button = ttk.Button(self.ovo_label_frame, text="Volcano plot", command=lambda: self.plot_ovo_volcano())
        self.ovo_volcano_plot_button.pack(side='left', anchor='sw', padx=10, pady=10)
        self.ovo_volcano_plot_button.config(state=tk.DISABLED)

        self.ovo_volcano_plot_button_log2 = ttk.Button(self.ovo_label_frame, text=f"Volcano log2plot", command=lambda: self.plot_ovo_volcano_log2())
        self.ovo_volcano_plot_button_log2.pack(side='left', padx=10, pady=10, anchor='sw')
        self.ovo_volcano_plot_button_log2.config(state=tk.DISABLED)

        step_index=2
        self.continue_button_three = ttk.Button(self.ovo_label_frame, text="Continue",
                                     command=lambda idx=step_index: self.complete_step(step_index))
        self.continue_button_three.pack(side='left', padx=10, pady=10, anchor='s')
        self.continue_button_three.config(state=tk.DISABLED)

        self.heat_frame = tk.Frame(frame)
        self.heat_frame.grid(row=1, column=0, sticky='nsew')
        self.heat_toolbar_frame = tk.Frame(frame)
        self.heat_toolbar_frame.grid(row=2, column=0, sticky='nsew')

        self.ovo_score_frame = tk.Frame(frame)
        self.ovo_score_frame.grid(row=1, column=1, sticky='nsew')
        self.ovo_score_toolbar_frame = tk.Frame(frame)
        self.ovo_score_toolbar_frame.grid(row=2, column=1, sticky='nsew')

        self.heatmap_canvas = FigureCanvasTkAgg(self.heat_fig, master=self.heat_frame)
        self.heatmap_canvas.get_tk_widget().pack(side='top', fill='both', expand=1, anchor='nw', padx=10)
        heatmap_toolbar = NavigationToolbar2Tk(self.heatmap_canvas, self.heat_toolbar_frame)
        heatmap_toolbar.update()
        heatmap_toolbar.pack(side='bottom', fill='x')

        self.hs_canvas = FigureCanvasTkAgg(self.hs_fig, master=self.ovo_score_frame)
        self.hs_canvas.get_tk_widget().pack(side='top', fill='both', expand=1, anchor='nw', padx=10)
        ovo_score_toolbar = NavigationToolbar2Tk(self.hs_canvas, self.ovo_score_toolbar_frame)
        ovo_score_toolbar.update()
        ovo_score_toolbar.pack(side='bottom', fill='x')

        label = ttk.Label(self.choose_fit_frame, text='Choose method to fit OPLS-models:')
        label.pack(anchor='w', pady=10)
        fit_methods = [('Autofit', 'af'), ('Max Q2', 'max_q2'), ('Choose number of components', 'choose'), ('Minimize CV-ANOVA', 'min_cv')]
        self.fit_option = tk.StringVar()

        
        for val in fit_methods:
            method, m = val
            button = ttk.Radiobutton(self.choose_fit_frame, text=f"{method}", variable=self.fit_option, value=m, command=lambda value=m: self.on_fit_option_change(value))
            button.pack(anchor='w', padx=10)
        if self.how_fit:
            self.fit_option.set(self.how_fit)
            self.on_fit_option_change(self.fit_option.get())
        else:
            self.fit_option.set('af')
        label = ttk.Label(self.choose_scale_frame, text='Choose scaling:')
        label.pack(anchor='w', pady=10)
        scale_methods = [('UV', 'uv'), ('Ctr', 'ctr'), ('None', 'none')]
        self.scale_option = tk.StringVar()
        if self.how_scale:
            self.scale_option.set(self.how_scale)
        else:
            self.scale_option.set('uv')
        for val in scale_methods:
            method, m = val
            button = ttk.Radiobutton(self.choose_scale_frame, text=f"{method}", variable=self.scale_option, value=m)
            button.pack(anchor='w', padx=10)

        self.confirm_button_fit = ttk.Button(self.heat_label_frame, text='Confirm and build one vs one models', command=lambda : self.build_ovo_models(self.heat_frame))
        self.confirm_button_fit.pack(side='right', anchor='se', padx=10, pady=10)
        if self.distance_matrix is not None and check_ovo_models(self.ovo_models):
            self.model_map = self.create_model_map(len(self.distance_matrix), self.ovo_models)
            self.heatmap = sns.heatmap(self.distance_matrix, linewidth=0.5, ax=self.heat_ax, cbar=False, xticklabels=self.train_df[self.class_col].unique(), yticklabels=self.train_df[self.class_col].unique(), annot=True)
            self.heatmap_canvas.draw()
            self.heat_fig.canvas.mpl_connect("button_press_event", self.button_click)
            self.heat_ax.set_aspect('auto')
            self.heat_fig.tight_layout()
            self.continue_button_three.config(state=tk.NORMAL)



    def on_fit_option_change(self, selected_fit_option):
    # Clear only the dynamic frame
        self.clear_frame(self.fit_dynamic_frame)
        if selected_fit_option == "choose":
            label = ttk.Label(self.fit_dynamic_frame, text="Choose number of components:")
            label.pack(pady=10, padx=10, anchor='nw')
            self.selected_components = tk.StringVar()
            
            self.column_label.pack(side="top",pady=10, padx=10, anchor='nw')
            self.selected_column = tk.StringVar()
            if self.n_components:
                self.selected_components.set(self.n_components)
            self.component_combobox = ttk.Combobox(self.fit_dynamic_frame, values=[i for i in range(1, 16)], textvariable=self.selected_components, state="readonly")
            self.component_combobox.pack(side="top",pady=10, padx=10, anchor='nw')
            self.component_combobox.bind("<<ComboboxSelected>>", self.on_component_selected)
    
    def on_component_selected(self, event):
        self.n_components = int(self.selected_components.get())
        
    def build_ovo_models(self, frame):
        self.confirm_button_fit.config(state=tk.DISABLED)
        clear_ovo_models()
        clear_bin_models()
        self.how_fit = self.fit_option.get()
        self.how_scale = self.scale_option.get()
        excl_test_data = self.entries[1].get('test_df_option', None)
        class_col = self.class_col
        classes = self.train_df[class_col].unique()
        num_classes = len(classes)
        D = np.zeros((num_classes, num_classes))

        sns.heatmap(D, linewidth=0.5, ax=self.heat_ax, cbar=False, xticklabels=False, yticklabels=False)
        self.heatmap_canvas.draw()
        self.heat_fig.tight_layout()
        def compute_and_update():
            for D, Dnames, models in one_vs_one_calc(self.how_fit, self.how_scale, self.df_names[self.entries[0]['dataframe']], self.test_df.index, self.train_df, class_col, excl_test_data, n_components=self.n_components):
                # Update the heatmap with new values
                self.heat_ax.clear()
                sns.heatmap(D, linewidth=0.5, ax=self.heat_ax, cbar=False, xticklabels=False, yticklabels=False)
                self.heatmap_canvas.draw()
                # Force update the GUI to show progress
                frame.update_idletasks()
            return D, Dnames, models
        D, Dnames, models = compute_and_update()
        self.heatmap = sns.heatmap(D, linewidth=0.5, ax=self.heat_ax, cbar=False, xticklabels=classes, yticklabels=classes, annot=True)
        self.distance_matrix = D.copy()
        self.ovo_models = models
        self.model_map = self.create_model_map(len(self.distance_matrix), self.ovo_models)

        self.heat_ax.set_aspect('auto')
        self.heat_fig.tight_layout()
        self.heatmap_canvas.draw()

        self.heat_fig.canvas.mpl_connect("button_press_event", self.button_click)
        self.confirm_button_fit.config(state=tk.NORMAL)
        self.continue_button_three.config(state=tk.NORMAL)
        self.entries[2].update({'ovo_models' : self.ovo_models})
        self.entries[2].update({'how_fit': self.how_fit})
        self.entries[2].update({'how_scale': self.how_scale})
        self.look_for_change(int(self.previous_tab_index))
        self.entries[2].update({'distance_matrix' : self.distance_matrix})
        self.Dnames = Dnames
        self.entries[2].update({'Dnames' : self.Dnames})
        self.entries[2].update({'n_components': self.n_components})

    def create_model_map(self, size, ovo_models):
        model_map = {}
        index = 0
        for row in range(size):
            for col in range(row + 1, size):  # Only map upper triangular cells
                model_map[(row, col)] = ovo_models[index]
                model_map[(col, row)] = ovo_models[index]
                index += 1
        return model_map

    def plot_ovo_scores(self):
        self.hs_ax.clear()
        plot_scores(self.model_number, self.hs_ax, self.hs_fig, self.train_df, self.class_col)
        self.hs_canvas.draw()

    def plot_ovo_volcano(self):
        self.hs_ax.clear()
        plot_volcano(self.model_number, self.hs_ax, self.hs_fig, self.train_df, self.class_col)
        self.hs_canvas.draw()

    def plot_ovo_volcano_log2(self):
        self.hs_ax.clear()
        plot_volcano_log2(self.model_number, self.hs_ax, self.hs_fig, self.train_df, self.class_col)
        self.hs_canvas.draw()

    def button_click(self, event):
        if event.inaxes == self.heat_ax:
            x_data, y_data = self.heat_ax.transData.inverted().transform((event.x, event.y))
            self.ovo_score_plot_button.config(state=tk.NORMAL)
            self.ovo_volcano_plot_button.config(state=tk.NORMAL)
            self.ovo_volcano_plot_button_log2.config(state=tk.NORMAL)
            for i, p in enumerate(self.heatmap.collections[0].get_paths()):
                # if p.contains_point((event.x, event.y)):
                if p.contains_point((x_data, y_data)):
                    # Get the index of the cell
                    row, col = self.get_cell_index(p)
                    if not row == col:
                        self.model_number = self.model_map[(row, col)]
                        value = self.distance_matrix[row, col]
                        self.plot_ovo_scores()
                    break
            
    def get_cell_index(self, path):
        """
        Given a path, return the corresponding row and column index.
        """
        # Get the bounding box of the path
        bbox = path.get_extents()
        
        # Calculate the row and column index from the bounding box
        row = int((bbox.y0 + bbox.y1) / 2)
        col = int((bbox.x0 + bbox.x1) / 2)
        return row, col

    def create_label_and_entry(self, frame, text):
        label = ttk.Label(frame, text=text)
        label.pack(pady=10)
        entry_text = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=entry_text)
        entry.pack(pady=10)

    def add_complete_button(self, frame, step_index):
        complete_button = ttk.Button(frame, text=f"Complete Step {step_index}",
                                     command=lambda idx=step_index-1: self.complete_step(step_index-1))
        complete_button.pack(side='left', padx=10, pady=10, anchor='nw')

    def add_continue_button(self, frame, step_index):
        continue_button = ttk.Button(frame, text="Continue",
                                     command=lambda idx=step_index-1: self.complete_step(step_index-1))
        continue_button.pack(side='left', padx=10, pady=10, anchor='s')

    def create_linkage_selection(self, frame):
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        self.link_label_frame = tk.Frame(frame)
        self.link_label_frame.grid(row=0, column=0, sticky='nsew')

        self.score_label_frame = tk.Frame(frame)
        self.score_label_frame.grid(row=0, column=1, sticky='nsew')

        self.link_frame = tk.Frame(frame)
        self.link_frame.grid(row=1, column=0, sticky='nsew')
        self.link_toolbar_frame = tk.Frame(frame)
        self.link_toolbar_frame.grid(row=2, column=0, sticky='nsew')

        self.score_frame = tk.Frame(frame)
        self.score_frame.grid(row=1, column=1, sticky='nsew')
        self.score_toolbar_frame = tk.Frame(frame)
        self.score_toolbar_frame.grid(row=2, column=1, sticky='nsew')

        self.score_canvas = FigureCanvasTkAgg(self.score_fig, master=self.score_frame)
        self.score_canvas.get_tk_widget().pack(side='top', pady=10, padx=10, anchor='sw', fill='both', expand=True)
        self.score_fig.tight_layout()
        score_toolbar = NavigationToolbar2Tk(self.score_canvas, self.score_toolbar_frame)
        score_toolbar.update()
        score_toolbar.pack(side='bottom', fill='x')


        label = ttk.Label(self.link_label_frame, text="Select linkage type to visualize the dendrogram:")
        label.pack(anchor='w', pady=5, padx=10)

        self.plot_buttons = []

        self.linkage_var = tk.StringVar(value=None)
        complete_button = ttk.Button(self.link_label_frame, text="Complete", command=lambda: self.update_dendrogram("complete"))
        complete_button.pack(side="left", padx=10, pady=5, anchor='nw')

        average_button = ttk.Button(self.link_label_frame, text="Average", command=lambda: self.update_dendrogram("average"))
        average_button.pack(side="left", padx=10, pady=5, anchor='nw')

        ward_button = ttk.Button(self.link_label_frame, text="Ward", command=lambda: self.update_dendrogram("ward"))
        ward_button.pack(side="left", padx=10, pady=5, anchor='nw')

        single_button = ttk.Button(self.link_label_frame, text="Single", command=lambda: self.update_dendrogram("single"))
        single_button.pack(side="left", padx=10, pady=5, anchor='nw')        

        self.calc_bin_button = ttk.Button(self.link_label_frame, text=f"Build OPLS-HDA model",
                                command=lambda: self.calc_binary_models())
        self.calc_bin_button.pack(side='right', pady=5, padx=10, anchor='nw')
        self.calc_bin_button.config(state=tk.DISABLED)
        self.dendrogram_canvas = FigureCanvasTkAgg(self.fig, master=self.link_frame)
        self.dendrogram_canvas.get_tk_widget().pack(side='top', pady=10, padx=10, anchor='sw', fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.dendrogram_canvas, self.link_toolbar_frame)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')

        if self.link_choice and check_bin_models(self.bin_models):
            self.text_annotations = []
            self.update_dendrogram(self.link_choice)
            for i in range(0, len(self.incl_classes)-1):
                text = self.ax.text(self.x[len(self.x)-1-i], self.y[len(self.x)-1-i],''.join([" Split #",str(i+1)]),  fontsize=10, horizontalalignment="center", verticalalignment="center",bbox=dict(boxstyle="round", ec=(0.1, 0.6, 0.1), fc=(0.6, 0.8, 0.6),))
                self.text_annotations.append((text, i+1))
                self.dendrogram_canvas.draw()
                self.link_frame.update_idletasks()
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.ax.set_aspect('auto')
        self.fig.tight_layout()

    def update_dendrogram(self, linkage_type):
        self.linkage_var.set(linkage_type)
        if self.link_choice == linkage_type and check_bin_models(self.bin_models):
            self.calc_bin_button.config(state=tk.NORMAL)
            self.data = squareform(self.distance_matrix)
            self.linked = linkage(self.data, linkage_type)
            self.ax.clear()
            self.ax.set_title(f'Linkage type: {linkage_type}')
            dn = dendrogram(self.linked, labels=self.Dnames, ax=self.ax, leaf_rotation=90)

            X = dn.get("icoord")
            Y = dn.get("dcoord")
            x=np.zeros((len(X),1))
            y=np.zeros((len(Y),1))
            for i in range(0,len(X)):
                x[i]=np.mean(X[i])
                y[i]=np.max(Y[i])
            x=np.squeeze(x)
            y=np.squeeze(y)

            i=np.argsort(y)
            x=x[i]
            y=y[i]
            self.x = x
            self.y = y
            self.text_annotations = []
            for i in range(0, len(self.incl_classes)-1):
                text = self.ax.text(self.x[len(self.x)-1-i], self.y[len(self.x)-1-i],''.join([" Split #",str(i+1)]),  fontsize=10, horizontalalignment="center", verticalalignment="center",bbox=dict(boxstyle="round", ec=(0.1, 0.6, 0.1), fc=(0.6, 0.8, 0.6),))
                self.text_annotations.append((text, i+1))
                self.dendrogram_canvas.draw()
                self.link_frame.update_idletasks()
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.ax.set_aspect('auto')
            self.fig.tight_layout()
        elif linkage_type:
            self.calc_bin_button.config(state=tk.NORMAL)
            self.data = squareform(self.distance_matrix)
            self.linked = linkage(self.data, linkage_type)
            self.ax.clear()
            self.ax.set_title(f'Linkage type: {linkage_type}')
            dn = dendrogram(self.linked, labels=self.Dnames, ax=self.ax, leaf_rotation=90)

            X = dn.get("icoord")
            Y = dn.get("dcoord")
            x=np.zeros((len(X),1))
            y=np.zeros((len(Y),1))
            for i in range(0,len(X)):
                x[i]=np.mean(X[i])
                y[i]=np.max(Y[i])
            x=np.squeeze(x)
            y=np.squeeze(y)

            i=np.argsort(y)
            x=x[i]
            y=y[i]
            self.x = x
            self.y = y

            for i in range(0,len(x)):
                self.ax.text(x[i], y[i],''.join([" Split #",str(len(x)-i)]),  fontsize=10, horizontalalignment="center", verticalalignment="center",bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))

            self.ax.scatter(x, y,s=80,c="k")
            self.ax.set_ylabel("Distance: Cohen's d based on YpredCV", fontsize=12)
            self.ax.set_xlabel("Classes", fontsize=12)
            self.ax.set_aspect('auto')
            self.fig.tight_layout()
            self.dendrogram_canvas.draw()
        else:
            pass

    def calc_binary_models(self):
        # self.linkage_var.set(linkage_type)
        if self.link_choice != self.linkage_var.get():
            self.bin_models = []
        self.link_choice = self.linkage_var.get()
        self.entries[3].update({'linkage_choice' : self.link_choice})  # Add the linkage_var to entries
        self.look_for_change(int(self.previous_tab_index))
        if not check_bin_models(self.bin_models) or self.how_fit != self.entries[2].get('how_fit', None):
            clear_bin_models()
            class_col = self.class_col
            excl_test_data = self.entries[1].get('test_df_option', None)
            self.bin_models = []
            self.text_annotations = []
            for model, i in bin_models_calc(self.fit_option.get(), self.how_scale, self.linked, dataset_name=self.df_names[self.entries[0]['dataframe']], excl_from_training_set=self.test_df.index, train_df=self.train_df, class_col=class_col, Dnames=self.Dnames, excl_test_data=excl_test_data, n_components=self.n_components):
                text = self.ax.text(self.x[len(self.x)-1-i], self.y[len(self.x)-1-i],''.join([" Split #",str(i+1)]),  fontsize=10, horizontalalignment="center", verticalalignment="center",bbox=dict(boxstyle="round", ec=(0.1, 0.6, 0.1), fc=(0.6, 0.8, 0.6),))
                self.text_annotations.append((text, i+1))
                self.bin_models.append(model[0])
                self.dendrogram_canvas.draw()
                self.link_frame.update_idletasks()
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.entries[3].update({'bin_models' : self.bin_models})
        else:
            self.text_annotations = []
            for i in range(0, len(self.incl_classes)-1):
                text = self.ax.text(self.x[len(self.x)-1-i], self.y[len(self.x)-1-i],''.join([" Split #",str(i+1)]),  fontsize=10, horizontalalignment="center", verticalalignment="center",bbox=dict(boxstyle="round", ec=(0.1, 0.6, 0.1), fc=(0.6, 0.8, 0.6),))
                self.text_annotations.append((text, i+1))
                self.dendrogram_canvas.draw()
                self.link_frame.update_idletasks()
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        for text, split_n in self.text_annotations:
            bbox = text.get_window_extent(self.fig.canvas.get_renderer())
            if bbox.contains(event.x, event.y):
                self.split_click(split_n)
                break

    def split_click(self, split_n):
        if self.plot_buttons:
            for b in self.plot_buttons:
                b.destroy()
        score_plot_button = ttk.Button(self.score_label_frame, text=f"CV-Score plot", command=lambda: split_plot_scores(split_n))
        score_plot_button.pack(side='left', padx=10, pady=5, anchor='sw')
        self.plot_buttons.append(score_plot_button)
        volcano_plot_button = ttk.Button(self.score_label_frame, text=f"Volcano plot", command=lambda: split_plot_volcano(split_n))
        volcano_plot_button.pack(side='left', padx=10, pady=5, anchor='sw')
        self.plot_buttons.append(volcano_plot_button)
        volcano_plot_button_log2 = ttk.Button(self.score_label_frame, text=f"Volcano log2plot", command=lambda: split_plot_volcano_log2(split_n))
        volcano_plot_button_log2.pack(side='left', padx=10, pady=5, anchor='sw')
        self.plot_buttons.append(volcano_plot_button_log2)
        train_cf_matrix_button = ttk.Button(self.score_label_frame, text=f"Train CV-confusion matrix", command=lambda: split_plot_cv_conf_matrix())
        train_cf_matrix_button.pack(side='left', padx=10, pady=5, anchor='sw')
        self.plot_buttons.append(train_cf_matrix_button)
        # if len(self.test_df.index) != 0:
        test_cf_matrix_button = ttk.Button(self.score_label_frame, text=f"Test confusion matrix", command=lambda: split_plot_test_conf_matrix())
        test_cf_matrix_button.pack(side='left', padx=10, pady=5, anchor='sw')
        self.plot_buttons.append(test_cf_matrix_button)
        if len(self.test_df.index) == 0:
            test_cf_matrix_button.config(state=tk.DISABLED)
        elif len(self.test_df.index) == len(self.train_df.index):
            if all(self.test_df.index == self.train_df.index):
                test_cf_matrix_button.config(state=tk.DISABLED)
        else:
            test_cf_matrix_button.config(state=tk.NORMAL)

        ## plot functions
        def split_plot_scores(split_n):
            self.score_ax.clear()
            plot_scores(self.bin_models[split_n-1], self.score_ax, self.score_fig, self.train_df, self.class_col)
            self.score_canvas.draw()

        def split_plot_volcano(split_n):
            self.score_ax.clear()
            plot_volcano(self.bin_models[split_n-1], self.score_ax, self.score_fig, self.train_df, self.class_col)
            self.score_canvas.draw()

        def split_plot_volcano_log2(split_n):
            self.score_ax.clear()
            plot_volcano_log2(self.bin_models[split_n-1], self.score_ax, self.score_fig, self.train_df, self.class_col)
            self.score_canvas.draw()

        def split_plot_cv_conf_matrix():
            self.score_ax.clear()
            plot_cv_conf_matrix(bin_models=self.bin_models, dataset_name=self.df_names[self.entries[0]['dataframe']],linked_data=self.linked, link_option=self.link_choice, excl_from_training_set=self.test_df.index, train_df=self.train_df, class_col=self.class_col, Dnames=self.Dnames, excl_test_data=self.entries[1].get('test_df_option', None), ax=self.score_ax, fig=self.score_fig)
            self.score_canvas.draw()

        def split_plot_test_conf_matrix():
            self.score_ax.clear()
            plot_test_conf_matrix(bin_models=self.bin_models, dataset_name=self.df_names[self.entries[1]['test_df_idx']], train_dataset_name=self.df_names[self.entries[0]['dataframe']], linked_data=self.linked, link_option=self.link_choice, excl_from_test_set=self.train_df.index, train_df=self.train_df, test_df=self.test_df, class_col=self.class_col, Dnames=self.Dnames, ax=self.score_ax, fig=self.score_fig, excl_test_data=self.entries[1].get('test_df_option', None), excl_from_training_set=self.test_df.index)
            self.score_canvas.draw()
        ## Plot scores on split_click
        split_plot_scores(split_n)

    def deep_compare(self, value1, value2):
        # Check for numpy arrays and convert them to lists for comparison
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            return np.array_equal(value1, value2)
        
        # Check for pandas DataFrames and use equals for comparison
        if isinstance(value1, pd.DataFrame) and isinstance(value2, pd.DataFrame):
            return value1.equals(value2)

        # Handle lists, including nested structures
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            for v1, v2 in zip(sorted(value1), sorted(value2)):
                if not self.deep_compare(v1, v2):
                    return False
            return True
        
        # Direct comparison for other types
        return value1 == value2

    def compare_dicts_with_lists(self, dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        
        for key in dict1:
            if not self.deep_compare(dict1[key], dict2[key]):
                return False
        return True

    def on_tab_change(self, event):
        selected_tab = self.notebook.index("current")
        self.previous_tab_index = int(selected_tab)
        self.entries[0].update({'prev_idx': int(selected_tab)})
        self.entries[0].update({'curr_idx': int(selected_tab)})
        self.original_values = self.deep_copy_dict(self.entries)

    def deep_copy_dict_inside(self, d_dict):
        copied_dict = {}
        for key, value in d_dict.items():
            copied_dict[key] = value
        return copied_dict
    
    def deep_copy_dict(self, d_list):
        copied_list = []
        for d in d_list:
            copied_dict = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    copied_dict[key] = self.deep_copy_dict_inside(value)
                elif isinstance(value, list):
                    copied_dict[key] = [self.deep_copy_dict(item) if isinstance(item, dict) else item for item in value]
                elif isinstance(value, pd.DataFrame):
                    copied_dict[key] = value.copy(deep=True)
                else:
                    copied_dict[key] = copy.deepcopy(value)
            copied_list.append(copied_dict)
        return copied_list

    def look_for_change(self, step_idx):
        print(f"Looking for change in step {step_idx + 1}")
        for dict1, dict2 in zip(self.entries, self.original_values):
            if not self.compare_dicts_with_lists(dict1, dict2):
                print("Original value changed, resetting subsequent steps")
                self.reset_subsequent_steps(step_idx)
                self.original_values = self.deep_copy_dict(self.entries)
                break
            else:
                print("Nothing has changed")

    def complete_step(self, step_idx):
        for dict1, dict2 in zip(self.entries, self.original_values):
            if not self.compare_dicts_with_lists(dict1, dict2):
                self.reset_subsequent_steps(step_idx)
                if step_idx + 1 < self.num_steps:
                    self.notebook.tab(step_idx + 1, state='normal')
                    self.notebook.select(step_idx + 1)
                break
            else:
                self.entries[step_idx + 1].update({'tab_state': 'normal'})
                self.notebook.tab(step_idx + 1, state='normal')
                self.notebook.select(step_idx + 1)
                break
        self.previous_tab_index = step_idx
        self.entries[0].update({'prev_idx': step_idx})
        self.entries[0].update({'curr_idx': step_idx+1})
        self.original_values = self.deep_copy_dict(self.entries)

    def clear_ovo_step(self):
        ## Clear OvO-step
        self.entries[2].clear()
        self.distance_matrix = None
        self.heat_ax.clear()
        self.continue_button_three.config(state=tk.DISABLED)
        if hasattr(self, 'heatmap_canvas'):
            self.heatmap_canvas.draw()
        self.hs_ax.clear()
        if hasattr(self, 'hs_canvas'):
            self.hs_canvas.draw()
        clear_ovo_models()

    def clear_dendro_hda_step(self):
        self.entries[3].clear()
        clear_bin_models()
        self.calc_bin_button.config(state=tk.DISABLED)
        self.ax.clear()
        if hasattr(self, 'dendrogram_canvas'):
            self.dendrogram_canvas.draw()
        self.score_ax.clear()
        if hasattr(self, 'score_canvas'):
            self.score_canvas.draw()

    def reset_subsequent_steps(self, start_idx):
        ### If training data is changed
        if start_idx == 0:
            self.clear_ovo_step()
            self.clear_dendro_hda_step()
        ### If training data is changed in test-set step
        elif start_idx == 1:
            if not self.entries[1].get('train_df', None).equals(self.original_values[1].get('train_df', None)):
                print('Training data changed')
                self.clear_ovo_step()
                self.clear_dendro_hda_step()                
        ### If something changes in OvO-step
        elif start_idx == 2:
            ## Clear Link-HDA step
            self.clear_dendro_hda_step()
        for i in range(start_idx + 1, self.num_steps):
            self.notebook.tab(i, state='disabled')
            self.entries[i].update({'tab_state': 'disabled'})

    def on_closing(self):
        self.storage['opls_hda_entries'] = self.entries
        self.quit()
        plt.close('all')
        self.destroy()

# Run the application
df_dict = concat_dfs_from_simca()
project = umetrics.SimcaApp.get_active_project()
storage = project.get_project_storage()
app = SequentialTabsGUI(dfs=df_dict)
app.mainloop()

### access names if pandas df is broken
# proj.data_builder().create("ObsID", model=22).get_value_ids().get_names(6)
## Exportera data
