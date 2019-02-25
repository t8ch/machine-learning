# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "69bbf41d-1a30-4ea1-9bcc-41a8c52de069"}, "cell_type": "markdown"}
# # Random Forests and Ensembles
#
# <center>a brief introduction</center>

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "2e1dd02f-f9b9-4cb8-a283-5efd45fcbb87"}, "cell_type": "markdown"}
# ## model averaging and stacking
# - "Stacking [..] involves training a learning algorithm to combine the predictions of several other learning algorithms." (wiki)
# - Bayesian **averaging** of models $M_m$ (training data $Z$):
# $$E(\text{pred}|Z) = \sum_m E(\text{pred}|Z,M_m)p(M_m|Z)$$
# - **stacking**: find best superposition of predictions from single learners (here regression)
# $$\hat w^{st} = \text{argmin}_{w} \sum_i \left[ y_i - \sum_m w_m \hat f_m^{-i}(x_i)\right]^2 ,$$
# where $f_m^{-i}(x)$ is the prediction of $x$, using model $m$, applied to data set with $x_i$ removed. 
# - related to leave-one-out

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "e39941e8-a962-45ec-b70b-876384093f44"}, "cell_type": "markdown"}
# ## bootstrapping
# -  generate distinct data sets by repeatedly sampling observations from the original data set with replacement
# - leverage learning for a given data set

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "9f98f44c-dcf5-48dd-995d-ae683013ea80"}, "cell_type": "markdown"}
# ## bagging
# - bagging = "bootstrap aggregating"
# - averaging the predictions $\hat f^{*b}$ of many learners (each with high variance) that were trained on bootstrap samples
#
# $$\hat f_\mathrm{bag} = \frac{1}{B}\sum_{b=1}^B \hat f^{*b}(x)$$

# + {"slideshow": {"slide_type": "subslide"}, "nbpresent": {"id": "6802077c-890a-47fa-8440-316eb1e4cd37"}, "cell_type": "markdown"}
# ### bagged trees
# - grow $B$ deep decision trees (on bootstrapped data) that have high variance but low bias
# - averaging reduces variance ($\propto 1/n$)

# + {"slideshow": {"slide_type": "subslide"}, "nbpresent": {"id": "eec641bd-d7ac-40d8-896a-eb974e0db21d"}, "cell_type": "markdown"}
# <img src="figures/bagging1.jpg" width="400">

# + {"slideshow": {"slide_type": "subslide"}, "nbpresent": {"id": "6ee91354-0841-43e6-be52-e096a3383bc9"}, "cell_type": "markdown"}
# <img src="figures/bagging2.jpg" width="550">

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "73ff84a1-75e6-4536-8062-fca89223d4b8"}, "cell_type": "markdown"}
# ## Random forests
# - similar to bagged trees but with less correlation among trees $\Rightarrow$ less variance
# - at each split only $m$ out of $p$ features are chosen for selection (typically $\sqrt{p}$)
# - trees become less similar as they cannot generally make the same split decisions
# <img src="figures/randomforest1.jpg" width="400">

# + {"slideshow": {"slide_type": "subslide"}, "nbpresent": {"id": "409709ff-6b38-48e4-85e4-8310553fd6e9"}, "cell_type": "markdown"}
# ### feature importance
# - **reminder**: trees are fitted by choosing splits that minimize
#     1. *Gini index* $G=\sum_k p_{mk}(1-p_{mk})$ or,
#     2. *Entropy* $D=-\sum_k p_{mk}\log(p_{mk})$,
#   <br>where $p_{mk}$ is the fraction of labels $k$ in region $m$
# - feature importance of feature $x$ is measured as the "summed reduction on the split criterion ($G$ or $E$) for all splits on $x$" 
# - feature randomization "equalizes" feature importances

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "d6e8e507-2919-47ad-986f-b342c155e400"}, "cell_type": "markdown"}
# ## Boosting and boosted trees

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "e8a87f44-aa91-4db6-8f67-ae830a5dab5b"}, "cell_type": "markdown"}
# ## Boosting methods
# - another very powerful variant of *committee-based* methods (not only for trees)
# - **basic idea**: fit "weak learners" to the residuals of previous weak learners
# - AdaBoost algorithm: final prediction $G(x)$ is given by $G(x)=f\left(\sum_{m=1}^M \alpha_mG_m(x)\right)$, where $f()$ depends on the classification/regression task and $\alpha_m$ is the weight of learner $G_m$.
# <img src="figures/AdaBoost-schematic.jpg" width="400">

# + {"slideshow": {"slide_type": "subslide"}, "nbpresent": {"id": "bd01da00-f8e0-4017-bbf6-1de03d3f7aba"}, "cell_type": "markdown"}
# current classifier $G_m$ is weighted according to their error; weights of each data point for the next classifier are calculated in 2d.
#
# <img src="figures/AdaBoost.jpg" width="450">

# + {"slideshow": {"slide_type": "slide"}, "nbpresent": {"id": "5a588d68-8705-413e-9b55-92fb6c4e9eee"}, "cell_type": "markdown"}
# ## Boosted trees
# tuning parameters:
# - number of trees $B$
# - shrinkage (learning rate) $\lambda$
# - number of splits $d$ in each tree
#
# <img src="figures/BoostedTrees-algo.jpg" width="450">

# + {"slideshow": {"slide_type": "subslide"}, "nbpresent": {"id": "df9bc7aa-da8e-493b-9318-1c2c375aef86"}, "cell_type": "markdown"}
# <img src="figures/randomforest-vs-boost.jpg" width="450">

# + {"nbpresent": {"id": "c5be4c4a-f233-40bb-8260-06803ead689f"}, "cell_type": "markdown"}
# ## Ensembles
# - "The idea of ensemble learning is to build a prediction model by combining the strengths of a collection of simpler base models." $\Rightarrow$ *all of the previous*
# - Ensemble learning can be broken down into two tasks: 
#     - developing a population of base learners from the training data,
#     - and then combining them to form the composite predictor
# - ESL provides more technical, algoritmic information (chapter 16)

# + {"slideshow": {"slide_type": "notes"}, "nbpresent": {"id": "dde58264-c806-473a-890c-ed178f3d8287"}}
!jupyter nbconvert RandomForests.ipynb --to slides --post serve --SlidesExporter.reveal_scroll=True 
