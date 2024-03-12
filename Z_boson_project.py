# -*- coding: utf-8 -*-

"""
PHYS20161 final assignment: Z_0 bozon

This code reads in data of the energy(Gev) ,cross-sectional area(nb)
and uncertainty in cross-sectional area(nb) in the form of a cvs file for
a z boson produced by a electron and a positron anhiliation. It outputs
the chi squared of the curve of best fit for the data and the mass, the
partial width and the lifetime of the z boson. This code also plots a
graph of the data along with the curve of best fit and a contour plot
of the chi squared of the curve of best fit.

Last Updated: 09/12/2021
Frantisek Bakota: 10618107
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc


# ~~~~~~~~~~~~~~~~~~~~~~~~~    GLOBAL    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DELIMITER = ","
TypeOfData = float
FILE_NAME1 = "z_boson_data_1.csv"
FILE_NAME2 = "z_boson_data_2.csv"
COMMENTS = "%"
CONVERSION = ((pc.hbar*pc.c)**2)/((1.6E-10)**2)*(10**37)
PARTIAL_WIDTH_EE = 83.91E-3
NUMBER_OF_DECIMALS = 4
COLLUM_MEASUREMENTS = 1
INITIAL_OUTLIER_MAX_NUM_STANDARD_DEV = 3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~  FUNCTION DEFINITION   ~~~~~~~~~~~~~~~~~~~~~~~~~~


def read_data(file_name):
    """
    reads in data
    Parameters
    ----------
    file_name : string
        Identifies the file to be opened.

    Returns
    -------
    array.

    """

    return np.genfromtxt(file_name, delimiter=DELIMITER,
                         dtype=TypeOfData, comments=COMMENTS,
                         invalid_raise=False)


def filter_data(data_array):
    """
    Takes a array of data and checks for nan and 0 values also
    uses a rudementary check for outliers.
    Parameters
    ----------
    data_array : numpy array
        unfiltered array of floats.

    Returns
    -------
    numpy array
        A filtered array of floats.

    """
    filtered_data = np.zeros((0, 3))
    for row in data_array:
        not_nan = True
        for number in row:
            if np.isnan(number) == False and np.around(
                    number, decimals=NUMBER_OF_DECIMALS) > 0.:
                pass
            else:
                not_nan = False

        if not_nan:
            temporary_array = np.array(row)
            filtered_data = np.vstack((filtered_data, temporary_array))
    return outlier_removal(filtered_data)


def function(parameters, energy):
    """
    Equation for the cross sectional area of a z_boson

    Parameters
    ----------
    parameters : numpy array
        Parameters that are to be fit to the experimental data.
    energy : numpy array
        DESCRIPTION.

    Returns
    -------
    numpy array
        Array of crossectional areas.

    """
    mass = parameters[0]
    partial_width = parameters[1]
    return CONVERSION*((12*np.pi/mass**2) *
                       energy**2/((energy**2-mass**2)**2 +
                                  (mass**2*partial_width**2)) *
                       PARTIAL_WIDTH_EE**2)


def chi_squared(parameters, data):
    "chi squared formula"
    chi_square = 0
    for row in data:
        chi_square += ((row[1] - function(parameters, row[0])) / row[2]) ** 2

    return chi_square


def plot(data, results):
    """
    Plots a errorbar graph for the experimental data and the curve of best fit
    Parameters
    ----------
    data : numpy array
        DESCRIPTION.
    results : numpy array
        results of fit.
    Returns
    -------
    None.

    """
    energy = np.linspace(start=np.min(data[:, 0]),
                         stop=np.max([data[:, 0]]), num=100)
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_xlabel("Energy in Gev")
    axis.set_ylabel("Cross sectional area " +
                    r'$\sigma_{{\mathrm{{Z_{{\mathrm{{0}}}}}}}}$' +
                    "in nb (nano barns)")
    axis.set_title("Graph of the " + r'$\chi^2_{{\mathrm{{min.}}}}$' +
                   "= {0:1.2f}".format(results[1]/len(data[:, 0])) +
                   " bosons crossectional area plotted "
                   "against the center of mass energy")
    axis.plot(energy, function(results[0], energy),
              label=("Curve of best fit where  " + "\n"
                     + r'$m_{{\mathrm{{Z_{{{\mathrm{0}}}.}}}}}$'
                     + " = {0:3.2f} GeV/c^2 & \n".format(results[0][0])
                     + r'$\Gamma_{{\mathrm{{Z_{{\mathrm{{0}}}}.}}}}$'
                     + "  = {0:3.2f} GeV. ".format(results[0][1])))
    axis.errorbar(data[:, 0], data[:, 1], data[:, 2], fmt="D",
                  label="data points with error bars")
    plt.grid()
    axis.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("Z_boson.png", dpi=400)
    plt.show()


def outlier_removal(data_array, curve_best_fit=None):
    """
    Remove outliers in two ways depending on the output if no curve_best_fit is
    inputted then the mean is used, but that only gets rid of extreme outliers,
    however if a curve_best_fit is inputted then all outliers can be removed.

    Parameters
    ----------
    data_array : numpy.ndarray
        DESCRIPTION.
    curve_best_fit : TYPE, optional
        curve of best fit of data. The default is None.

    Returns
    -------
    numpy.ndarray
        A numpy array without outliers.

    """
    standard_deviation = np.std(data_array[:, COLLUM_MEASUREMENTS])
    max_deviations = INITIAL_OUTLIER_MAX_NUM_STANDARD_DEV

    if isinstance(curve_best_fit, type(None)) is True:
        mean = np.average(data_array[:, COLLUM_MEASUREMENTS])
        distance_from_mean = abs(data_array[:, COLLUM_MEASUREMENTS] - mean)
        not_outlier = distance_from_mean < max_deviations*standard_deviation
    else:

        distance_from_model = abs(data_array[:, COLLUM_MEASUREMENTS] -
                                  curve_best_fit)

        not_outlier = distance_from_model < data_array[:, 2]*3

    return data_array[not_outlier]


def lifetime(partial_width, uncertainty):
    """
    Calculates the life time of a particle and its associated uncertainty
    Parameters
    ----------
    partial_width : float

    uncertainty : float
        uncertainty in the partial width.

    Returns
    -------
    float
        the lifetime of the z boson in seconds.

    """
    lifetime_ = pc.hbar/(partial_width * (1.6e-10))
    uncertainty_life = lifetime_ * uncertainty/partial_width
    return lifetime_, uncertainty_life


def mesh_arrays(x_array, y_array):
    """Takes two arrays of shape of the same shape
    and returns a mesh of the two arrays"""
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for _ in x_array:
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)
    return x_array_mesh, y_array_mesh


def find_uncertainty(array, sigma, minimum_chi, mesh):
    """
    Finds uncertainty of the chisquared if the contours are
    assumed to be circular.

    Parameters
    ----------
    array : numpy.ndarray
    sigma : float
        The standard deviation to be used to find the uncertainty.
    minimum_chi : float
    mesh : numpy.ndarray
        2D array of the parmeter with another parameter.

    Returns
    -------
    float
        the uncertainty in parameter (mesh) due to .

    """
    rounded_chi = np.round(minimum_chi, 3)
    array_chi = mesh[np.where(np.around(array, 3) == rounded_chi + sigma)]
    result_array = np.empty((0, len(array_chi)))
    for element in array_chi:
        result_array = np.vstack((result_array, array_chi-element))
    return np.max(result_array)/2


def contour_plot_and_uncertainty(results, data):
    """Plots a contour plot of the chi squared of the curve of best fit
    for the data against the mass and the partial width of the z boson"""

    a_array = np.linspace(-0.05+results[0][0], 0.05+results[0][0], 500)
    b_array = np.linspace(-0.05+results[0][1], 0.05+results[0][1], 500)
    a_mesh, b_mesh = mesh_arrays(a_array, b_array)

    contour_figure = plt.figure()
    axis = contour_figure.add_subplot(111)
    axis.scatter(results[0][0], results[0][1], marker="x", color="r")
    rounded_results = np.round(results[1], 3)
    contour_array = chi_squared((a_mesh, b_mesh), data)
    contourplot = axis.contour(a_mesh, b_mesh, contour_array,
                               levels=[rounded_results+1, rounded_results+2.3,
                                       rounded_results+5.99,
                                       rounded_results+9.21])
    labels = [r'$\chi^2_{{\mathrm{{min.}}}}$',
              r'$\chi^2_{{\mathrm{{min.}}}}+1.00$',
              r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',
              r'$\chi^2_{{\mathrm{{min.}}}}+5.99$',
              r'$\chi^2_{{\mathrm{{min.}}}}+9.21$']
    axis.clabel(contourplot, colors="r")

    for index, label in enumerate(labels):
        axis.collections[index].set_label(label)
    axis.grid()
    axis.set_xlabel(r'$m_{{\mathrm{{Z_{{{\mathrm{0}}}.}}}}}$'+" in GeV")
    axis.set_ylabel(r'$\Gamma_{{\mathrm{{Z_{{\mathrm{{0}}}}.}}}}$'+" in GeV")
    axis.set_title("Contour plot of the "+r'$\chi^2$'
                   + " of curves of fit for parameters "
                   + r'$m_{{\mathrm{{Z_{{{\mathrm{0}}}.}}}}}$'+" , "
                   + r'$\Gamma_{{\mathrm{{Z_{{\mathrm{{0}}}}.}}}}$')
    axis.legend(loc="upper right", bbox_to_anchor=(1.3, 0.8))
    plt.tight_layout()
    plt.savefig("Z_boson_chisquared_contour_plot.png", dpi=400)
    plt.show()

    return find_uncertainty(contour_array, 1, results[1],
                            a_mesh), find_uncertainty(contour_array,
                                                      1, results[1], b_mesh)


def main():
    """
    This function is the main part of the code

    Returns
    -------
    int
        1 if it runs correctly and 0 if a error
        occures when trying to open the files.

    """
    try:
        data1 = filter_data(read_data(FILE_NAME1))
        data2 = filter_data(read_data(FILE_NAME2))
        data3 = np.vstack((data1, data2))[np.argsort(np.vstack((data1,
                                                                data2))[:, 0])]

    except OSError:
        print("files not found")
        return 0

    results = fmin(lambda parameters: chi_squared(parameters, data3),
                   (90, 3), full_output=True, disp=False)
    data_final = outlier_removal(data3, function(results[0], data3[:, 0]))
    results_final = fmin(lambda parameters: chi_squared(parameters,
                                                        data_final),
                         results[0], full_output=True, disp=False)

    plot(data_final, results_final)
    mass_uncertainty, partial_width_uncertainty = contour_plot_and_uncertainty(
        results_final, data_final)
    print("Reduced chisquared has been found to be {0:3.3f}."
          " The mass of the z boson was found to be :"
          " {1:3.2f} ± {2:.2f} GeV/c^2 ""and the partial witdth of the z boson"
          " was found to be : "
          " {3:3.3f} ± {4:.3f}  GeV. ".format(results_final[1] /
                                              len(data_final[:, 0]),
                                              results_final[0][0],
                                              mass_uncertainty,
                                              results_final[0][1],
                                              partial_width_uncertainty))
    lifetime_and_uncertainty = lifetime(results_final[0][1],
                                        partial_width_uncertainty)
    print("The lifetime was of the z boson was found to be {0:.3} ± {1:.1} s".
          format(lifetime_and_uncertainty[0], lifetime_and_uncertainty[1]))

    return 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   MAIN CODE   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    main()
