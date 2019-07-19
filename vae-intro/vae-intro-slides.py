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

# + {"nbpresent": {"id": "69bbf41d-1a30-4ea1-9bcc-41a8c52de069"}, "slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Variational Autoencoders - an introduction

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Latent variables and generative models

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - **idea**: data is generated by a two-step (stochastic process)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - first, a low-dimensional and unobserved latent variable $z$ is determined (using $P(z)$)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - second, the observed data $x$ is generated via the decoder $P(x|z)$

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - examples for latent variables: digit between 0 and 9; presented sensory stimulus
# - examples for observed data: image of handwritten digit; neural activity 

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# $$P(x) = \int P(x|z)P(z) dz$$
#
# <center><img src="figs/gm1.jpg" width="300" align='middle'></center>

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# **generative models**
#
# if an appropriate representation of $z$ and a decoder $P(x|z)$ can be found, imaginary data samples can be generated by sampling from them
#
# most popular class of such systems: generative adversarial networks (GAN; *Goodfellow, Ian, et al. "Generative adversarial nets." NIPS, 2014.*)
#
# [example: artificially generated faces](http://thispersondoesnotexist.com)
#
# variational autoencoders (VAE) are another one...

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ## Variational Autoencoders
# ### Autoencoders

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - general scheme of autoencoders:
# $$x\longrightarrow z=f(x,\theta) \longrightarrow \hat x = g(z,\phi)$$
# - trying to optimize reconstruction $\hat x$ (for example by minimizing mean squared error)
# - latent representation $z$ with lower dimensionality $\Rightarrow$ compression/dimensionality reduction
# - for linear $f, g$: latent variables $z$ are projections on principal components

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# <center><img src="figs/mnist-pca.png" width="400" align='middle'></center>

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# ### Variational Autoencoders
# - VAE: $f$ and $g$ become probabilitic, non-deterministic functions that are implemented via neural networks 
# <center><img src="figs/vae1.png" width="400" align='middle'></center>[picture source](https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)
#
# - the latent space (hopefully) provides a nice low-dimensional manifold of the data
#
# <center>But how are the network parameters and the latent space found?</center>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### The variational bound

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - **problem**: the integral $p_\theta(x)=\int p_\theta (z)p_\theta(x|z)dz$ is intractable and cannot easily be maximized with respect to $\theta$ <br>(same for posterior $p_\theta(z|x)=p_\theta(x|z)p_\theta(z)/p_\theta(x)$)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - one solution: Monte Carlo sampling $p_\theta(x)\approx \frac{1}{N}\sum_i p(x|z_i)$, **but** this requires many sampling points for each $x$

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# - **variational inference**: approximate posterior $p_\theta(z|x)$ by parametrized function $q_\Phi(z|x)$, the (learnable) prior over $z$ by $p_\theta(z)$, and obtain variational **evidence lower bound** (ELBO):
# <center><img src="figs/elbo.jpg" width="630" align='middle'></center> <small>[Kingma & Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).]</small>
#
# - **interpretation**: lower bound wants the best possible tradeoff of
#   - KL divergence of posterior and prior (minimize)
#   - log-likelihood of the observed data (maximize)
# - interesting note on Eq. (2): negative of lower bound is variational free enegery as known from statistical physics<br>
# $$-\mathcal{L}=\mathbb{E}_q\left[E(x,z)\right]-H(q)$$

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# - **variational inference**: approximate posterior $p_\theta(z|x)$ by parametrized function $q_\Phi(z|x)$, the (learnable) prior over $z$ by p_\theta(z), and obtain variational **evidence lower bound** (ELBO):
# <center><img src="figs/elbo.jpg" width="630" align='middle'></center> <small>[Kingma & Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).]</small>
#
# - alternative interpretation 1: $\mathcal{L}$= optimized reconstruction + $\beta\cdot$regularization.  ($\beta$-VAE)
# - alternative interpretation 2: rate-distortion theory: $-\mathcal{L}=$distortion + (information) rate

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### (putative) advantages of random sampling from encoder
# - model is generative
# - latent space becomes smooth (data is denoised)

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# - **however**: "VAEs
# tend to generate blurry samples, a condition which has been
# attributed to using overly simplistic distributions for the
# prior" <small>[Gosh et al. From Variational to Deterministic Autoencoders.]</small>
#
# <center><figure>
# <img src="figs/kl-tradeoff.jpg" width="400" align='middle'>
# <figcaption>The tradeoff can be harmful for either reconstruction quality or sampling</figcaption>
# </figure></center>

# + {"slideshow": {"slide_type": "subslide"}, "cell_type": "markdown"}
# - rate-distortion curve characterizes the tradeoff between compression and reconstruction accuracy
# <center><img src="figs/rd.jpg" width="450" align='middle'></center> <small>[Alemi et al. "Fixing a broken ELBO." arXiv preprint arXiv:1711.00464 (2017).]</small>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### VAE in practice
# - often isotropic Gaussians are assumed as priors: $p_\theta(z)=\mathcal{N}(z;0,\mathbb{I})$
# - " The key is to notice that any distribution in d dimensions can
# be generated by taking a set of d variables that are normally distributed and
# mapping them through a sufficiently complicated function." <br><small>Doersch, Carl. "Tutorial on variational autoencoders." arXiv preprint arXiv:1606.05908 (2016).</small>
# - the KL divergence becomes analytic and does not have to be estimated
#
# <center><img src="figs/vae2.jpg" width="600" align='middle'></center><small>Doersch, Carl. "Tutorial on variational autoencoders." arXiv preprint arXiv:1606.05908 (2016).</small>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# These principles are the same as used in the LFADS system discussed last time.
#
# let's see some code in action...

# + {"nbpresent": {"id": "dde58264-c806-473a-890c-ed178f3d8287"}, "slideshow": {"slide_type": "notes"}}
!jupyter nbconvert vae-intro-slides.ipynb --to slides --post serve --SlidesExporter.reveal_scroll=True 