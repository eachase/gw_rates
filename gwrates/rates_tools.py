from __future__ import division
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfc, erfinv

__all__ = ['SampleCollection', 'ManyBackgroundCollection',
           'lnprob', 'lnprior', 'lnlike']

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

        for i in np.arange(number):
            # Draw a random number from a unifrom distribution
            rand = np.random.uniform()

            # Classify as background
            if rand < self.ratio_b:
                self.background.append(np.sqrt(2) * \
                    erfinv(1 - (1 - np.random.uniform()) * erfc(self.xmin / np.sqrt(2))))

            # Classify as foreground
            else:
                self.foreground.append(self.xmin * (1 - np.random.uniform())**(-1/3))


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

        samples = self.foreground + self.background
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
            `foreground_count`, `gaussian_background_count`, and
            `glitch_class_count` are set.

        Notes
        -----"""
        glitch_classes = kwargs.pop('glitch_classes', self.glitch_dict.keys())
        self.samples = {}

        # Draw foreground samples
        self.foreground_count = foreground_count
        self.samples['Foreground'] = self.xmin * (1 -
                                         np.random.uniform(size=foreground_count))**(-1/3)

        

        # Draw gaussian background samples
        self.gaussian_background_count = gaussian_background_count
        self.samples['Gaussian'] = np.sqrt(2) * erfinv(1 -
                                       (1 - np.random.uniform(
                                       size=gaussian_background_count))*
                                       erfc(self.xmin / np.sqrt(2)))

        # Define each glitch class to have SNRs defined in the glitch_dict
        for glitch_class in glitch_classes: 
            self.samples[glitch_class] = self.glitch_dict[glitch_class]

        self.glitch_class_count = sum([len(x) for x in self.samples.values()])


    def plot_hist(self):
        """
        Make a histogram of all drawn samples.
        """
        num_samples = self.foreground_count + self.gaussian_background_count + \
            self.glitch_class_count
        num_classes = len(self.samples.keys())
        num_bins = int(np.floor(np.sqrt(num_samples)))
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

        plt.figure(figsize=(20,10))

        for idx, icategory in enumerate(self.samples.keys()):
            plt.hist(self.samples[icategory], label=icategory,
                     color=colors[idx], bins=num_bins, cumulative=-1,
                     histtype='step')

        plt.legend(loc='upper right')
        plt.yscale('log', nonposy='clip')
        plt.xlim(0, None)
        plt.ylim(1, None)
        plt.xlabel('SNR')
        plt.ylabel('Number of Events  with SNR > Corresponding SNR')
        plt.title('%i Samples with Minimum SNR of %.2f' % (int(num_samples), self.xmin))
        plt.show()


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
        return np.sum(np.log(3 * xmin**3 * rate_f * samples**(-4) \
            + rate_b * (np.sqrt(np.pi/2) * erfc(
            xmin / np.sqrt(2)))**(-1) * np.exp(-samples**2 / 2)))
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
