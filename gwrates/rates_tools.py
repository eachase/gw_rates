from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfc, erfinv


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
