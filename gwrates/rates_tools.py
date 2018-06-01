from __future__ import division
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats
from scipy.special import erf, erfc, erfinv

__all__ = ['SampleCollection', 'ManyBackgroundCollection',
           'lnprob', 'lnprior', 'lnlike']

def power_law_pdf(x, k, xmin, xmax=np.inf):
    if k == -1:
        raise ValueError("k == -1 is handled by a logarithmic distribution.")
    assert xmin < x < xmax

    norm = (xmax**(k + 1) - xmin**(k + 1)) / (k+1)

    return x**k / norm

def power_law_rvs(k, xmin, xmax=np.inf, shape=1):
    if k == -1:
        raise ValueError("k == -1 is handled by a logarithmic distribution.")

    norm = (xmax**(k + 1) - xmin**(k + 1))

    x = np.random.uniform(0, 1, shape)

    return (x * norm + xmin**(k + 1))**(1. / (k + 1))

class SampleCollection(object):
    """
    A set of events with an associated ranking statistic
    """
    def __init__(self, rate_f_true, rate_b_true, xmin=0):
        """
        Initialize the sample collector

        Parameters:
        -----------
        rate_f_true: float
            known rate of foreground events

        rate_b_true: float
            known rate of background events

        xmin: float
            Minimum threshold SNR
        """

        self.ratio_b = rate_b_true / (rate_f_true + rate_b_true)
        self.foreground = []
        self.background = []
        self.xmin = xmin
        self.bins = None


    def draw_samples(self, number=1):
        """
        Draw either a foreground or background event, based on the R ratios

        Parameter:
        ----------
        number: int
            number of samples to draw
        """

        # Determine number of each category from Poissonian statistics
        n_fg, n_bg = stats.poisson.rvs(self.rate_f_true), \
                        stats.poisson.rvs(self.rate_b_true)

        # Background model is a chi^2 with 2 degrees of freedom
        self.background = stats.chi2.rvs(2, loc=self.xmin, size=n_bg)
        # Foreground model is a power law with index = -4
        self.foreground = power_law_rvs(-4, self.xmin, shape=n_fg)


    def plot_hist(self):
        """
        Make a histogram of all drawn samples.
        """

        num_samples = len(self.foreground) + len(self.background)
        num_bins = int(np.floor(np.sqrt(num_samples)))
        self.bins = np.linspace(0, max(np.max(self.background), np.max(self.foreground)),
                                num_bins)

        plt.figure()

        # Foreground histogram
        bin_counts, bins, _ = plt.hist(self.foreground, label='Foreground', alpha=0.5, bins=num_bins, cumulative=-1, color='purple')

        # Background histogram
        plt.hist(self.background, label='Background', alpha=0.5, bins=bins, cumulative=-1, color='0.75')

        plt.legend(loc='upper right')
        plt.yscale('log', nonposy='clip')
        plt.xlim(0, None)
        plt.ylim(1, None)
        plt.xlabel('SNR')
        plt.ylabel('Number of Events  with SNR > Corresponding SNR')
        plt.title('%i Samples with Minimum SNR of %.2f' % (int(num_samples), self.xmin))
        plt.show()

        print('Number of Foreground: ', len(self.foreground))
        print('Number of Backgruond:', len(self.background))


    def plot_cdf(self):
        """
        Make cumulative diagram
        """

        samples = np.concatenate((self.foreground, self.background))
        num_samples = len(samples)
        counts, bins = np.histogram(samples, bins=self.bins)
        cdf = np.cumsum(counts) / num_samples

        plt.figure()
        plt.plot(bins[:-1], cdf)
        plt.xlim(0, None)
        plt.ylim(0, None)
        plt.xlabel('SNR')
        plt.ylabel('Cumulative Number of Events')
        plt.title('CDF of %i Samples with Minimum SNR of %.2f' % (num_samples, self.xmin))
        plt.show()


class ManyBackgroundCollection(object):
    """
    A set of events with an associated ranking statistic.
    Events are divided into foreground and several different
    classes of background events.
    """
    def __init__(self, glitch_dict, xmin=3.5):
        """
        Initialize the sample collector

        Parameters:
        -----------
        glitch_dict: `dict`
            dictionary with glitch class as key (i.e. 'Scratchy')
            and corresponding snr time-series as value

        xmin: `float`, optional, default: 3.5
            Minimum threshold SNR

        Returns
        -------
        `ManyBackgroundCollection`: with the following
        attrs, `xmin`, `glitch_dict`

        Notes
        -----"""

        self.xmin = xmin
        self.glitch_dict = glitch_dict


    def draw_samples(self, foreground_count, gaussian_background_count,
                     **kwargs):
        """
        Draw a full set of foreground, background, and Gravity Spy
        events

        Parameters:
        -----------
        foreground_count : `int`
            known count of foreground events

        gaussian_background_count : `int`
            known count of background events

        glitch_classes : `list`, optional, default: `self.glitch_dict.keys()`
            if you would like
            to only populate samples from some of the gravityspy
            categories provide a list like `['Scratchy', 'Blip']`

        Returns
        -------
        self : `ManyBackgroundCollection` now has an attr
            `samples` that contains keys of 'Foreground', 'Gaussian'
            and list of glitch_classes. In addition, the attrs
            `foreground_count`, `gaussian_background_count`,
            `unlabeled_samples` and `num_samples` are set.

        Notes
        -----
        """

        glitch_classes = kwargs.pop('glitch_classes', self.glitch_dict.keys())
        self.samples = {}

        # Draw foreground samples
        self.foreground_count = foreground_count
        self.samples['Foreground'] = power_law_rvs(-4, self.xmin, \
                                            size=self.foreground_count)

        # Draw gaussian background samples
        self.gaussian_background_count = gaussian_background_count
        self.samples['Gaussian'] = stats.chi2.rvs(2, loc=self.xmin, \
                                size=self.gaussian_background_count)

        # Define each glitch class to have SNRs defined in the glitch_dict
        # FIXME: Note that this is only a shallow copy
        self.samples = self.glitch_dict.copy()

        # Create array of all samples, regardless of label
        self.unlabeled_samples = numpy.concatenate(self.samples.values())

        self.num_samples = len(self.unlabeled_samples)


    def plot_hist(self):
        """
        Make a histogram of all drawn samples.
        """
        num_classes = len(self.samples.keys())
        num_bins = int(np.floor(np.sqrt(self.num_samples)))
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

        # FIXME: need a robust and uniform way to define bins
        bins = np.linspace(self.xmin, 100, num_bins)

        plt.figure(figsize=(20,10))

        for idx, icategory in enumerate(self.samples.keys()):
            plt.hist(self.samples[icategory], label=icategory,
                     color=colors[idx], bins=bins, cumulative=-1,
                     histtype='step')

        plt.legend(loc='upper right')
        plt.yscale('log', nonposy='clip')
        plt.xlim(0, None)
        plt.ylim(1, None)
        plt.xlabel('SNR')
        plt.ylabel('Number of Events  with SNR > Corresponding SNR')
        plt.title('%i Samples with Minimum SNR of %.2f' % (int(self.num_samples), self.xmin))
        plt.show()


    def lnlike(self, counts, glitch_classes=[]):
        """
        Log Likelihood

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [foreground_counts, gaussian_counts, all_other_glitch_counts]
        """
        if np.all(counts >= 0):
            # Foreground likelihood
            fg_likelihood = getattr(self,  'Foreground' + '_evaluted')* \
                 counts[0]

            # Gaussian noise likelihood
            gauss_likelihood = getattr(self,  'Gaussian' + '_evaluted') * counts[1]

            # Likelihood for all other glitch sources of interest
            glitch_likelihood = 0
            for idx, iglitchtype in enumerate(glitch_classes):
                # Evaluate likelihood
                glitch_likelihood += counts[idx+2] * getattr(self, iglitchtype + '_evaluted')

            return np.sum(np.log(fg_likelihood + gauss_likelihood + \
                glitch_likelihood))
        else:
            return -np.inf


    def lnprior(self, counts):
        """
        Log Prior

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [foreground_counts, gaussian_counts, all_other_glitch_counts]

        N.B.: technically, the exp^(-Sum(counts)) term is part of the likelihood in FGMC
        """
        if np.all(counts >= 0):
            return -np.sum(counts) - 0.5*np.log(np.prod(counts))
        else:
            return -np.inf


    def lnprob(self, counts, glitch_classes=[]):
        """
        Combine log likelihood and log prior

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [foreground_counts, gaussian_counts, all_other_glitch_counts]
         """

        prior = self.lnprior(counts)
        posterior = self.lnlike(counts, glitch_classes)
        if not np.isfinite(prior):
            return -np.inf
        return prior + posterior



def lnlike(theta, samples, xmin):
    """
    Log Likelihood

    Parameters:
    -----------
    theta: iterable of two floats - (rate_f, rate_b)
        first entry corresponds to the rate of foreground events;
        second entry is the rate of background events;

    samples: array
        all SNR values in the given distribution

    xmin: float
        minimum threshold SNR
    """
    rate_f, rate_b = theta
    if rate_f > 0 and rate_b > 0:
        return np.sum(rate_f * power_law_pdf(samples, -4, self,xmin) \
            + rate_b * stats.chi2.pdf(2, loc=self.xmin))
    else:
        return -np.inf


def lnprior(theta):
    """
    Log Prior

    Parameters:
    -----------
    theta: iterable of two floats - (rate_f, rate_b)
        first entry corresponds to the rate of foreground events;
        second entry is the rate of background events;


    N.B.: technically, the exp^(-Rf - Rb) term is part of the likelihood in FGMC
    """

    rate_f, rate_b = theta
    if rate_f > 0 and rate_b > 0:
        return -rate_f - rate_b - 0.5*np.log(rate_f * rate_b)
    else:
        return -np.inf


def lnprob(theta, samples, xmin):
    """
    Combine log likelihood and log prior

    Parameters:
    -----------
    theta: iterable of two floats - (rate_f, rate_b)
        first entry corresponds to the rate of foreground events;
        second entry is the rate of background events;

    samples: array
        all SNR values in the given distribution

    xmin: float
        minimum threshold SNR
    """

    prior = lnprior(theta)
    posterior = lnlike(theta, samples, xmin)
    if not np.isfinite(prior):
        return -np.inf

    return prior + posterior


def compute_pastro(collection, lambda_post_samples, glitch_classes, glitch_kdes,
                   xmin=3.5, num_draw_from_lamda=1000, category='Foreground',
                   random_state=1986):
    """
    Calculate the probability that a given SNR sample
    comes from an astrophysical distribution.

    Parameters:
    -----------
    snr: float
        value of signal to noise ratio of interest

    post_samples: array of floats
        array of posterior samples of dimension
        (num_samples, num_source_classes) where
        num_samples is the number of posterior
        samples and num_source_classes is the number
        of classes considered (i.e. foreground, gaussian, etc.)

    source_classes: array of strings
        Each entry corresponds to the name of a type of
        source class: Foreground, Background, Blip, Scratchy, etc.

    glitch_kdes: dictionary
        Evaluated KDE for each GravitySpy glitch class.
        Keys are strings of GravitySpy class names.

    xmin: float
        Minimum threshold SNR

    num_iters: int
        number of samples drawn for Monte Carlo integration

    Returns:
    --------
    float or array
        P(astro) for each SNR value provided.
        This only returns one P(astro) if only one SNR is provided.

    """
    # Draw a sample from the posterior (some set of counts)
    counts = lambda_post_samples.sample(n=num_draw_from_lamda,
        random_state=random_state)

    # Compute the "likelihood ratio" for the drawn posterior sample
    setattr(collection, '{0}_likelihood'.format('Foreground'),
            np.multiply.outer(counts['Foreground'],
                              3 * xmin**3 * collection.unlabeled_samples**(-4)
                              ))

    setattr(collection, '{0}_likelihood'.format('Gaussian'),
            np.multiply.outer(counts['Gaussian'],
            (np.sqrt(np.pi/2) * erfc(
                xmin / np.sqrt(2)))**(-1) * np.exp(-collection.unlabeled_samples**2 / 2)
                                        ))

    # All GSpy glitches have an SNR greater than 7.5
    for idx, iglitchtype in enumerate(glitch_classes):
        setattr(collection, '{0}_likelihood'.format(iglitchtype),
            np.multiply.outer(counts[iglitchtype],
            np.exp(
            glitch_kdes[iglitchtype].score_samples(collection.unlabeled_samples.reshape(-1,1)))
            ))

    # Add to other samples
    denominator = 0
    for ikey in collection.samples.keys():
        denominator += getattr(collection, '{0}_likelihood'.format(ikey))

    likelihood_ratio = getattr(collection, '{0}_likelihood'.format(category))/\
        denominator

    # Report sum divided by N
    return likelihood_ratio.sum(axis=0) / num_draw_from_lamda
