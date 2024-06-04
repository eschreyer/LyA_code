import scipy.integrate as sp_int
import scipy.special as sp_special
import scipy.interpolate as sp_intpl
import numpy as np


class Gaussian2D():

#interacts with position, velocity of tail in functions of alpha, beta
#interacts with position, velocity of tail in functions of P_sw
#interacts with mlr

    def __init__(self, mdot, c_s, get_alpha, get_beta, get_zeta, get_PswD, D_interpolant_irreg):
        self.mdot = mdot
        self.c_s = c_s #can do variable c_s by making this a function
        self.get_zeta = get_zeta
        self.get_alpha = get_alpha
        self.get_beta = get_beta
        self.get_PswD = get_PswD
        self.D_interpolant_irreg = D_interpolant_irreg

    def __str__(self):
        return f'\u03C1 = \u03C1\u2080exp[-a\u00B2/\u03B1\u00B2 - b\u00B2/\u03B2\u00B2]'

    @classmethod
    def make_D_interpolant(self, is_zeta_zero = True):

        if is_zeta_zero == True: #we have an analytic expression for the mass loss rate

            def D_analytic_fnct_zeta0(log_mdot_d, zeta):

                return np.sqrt(np.log(np.exp(log_mdot_d) + 1))


            return D_analytic_fnct_zeta0

        else: #we need to make an interpolant

            def circular_cov_fnct2(R, r):
                """an alternative version of the circular coverage function that takes R as an array

                parameters
                 -----------------------

                R : array-like

                r: array-like
                """

                def integrand(t):

                    return np.exp(- t**2 * R**2 / 2) * sp_special.i0(t * R * r) * t

                y, err = sp_int.quad_vec(integrand, 0, 1)

                val = R**2 * np.exp(- r**2 /2) * y

                return val


            #evaluate mdot_d

            D_d = np.logspace(-2, 1, 1500)

            zeta = np.concatenate((-np.logspace(-2, 2, 1500), np.array([0])))

            D_d_mesh, zeta_mesh = np.meshgrid(D_d, zeta)

            A = D_d_mesh + np.sqrt(D_d_mesh**2 - zeta_mesh)

            B = D_d_mesh - np.sqrt(D_d_mesh**2 - zeta_mesh)

            log_mdot_d_mesh = D_d_mesh**2 + np.log(circular_cov_fnct2(A, B) - circular_cov_fnct2(B, A))  # mdot * cs^2 / u * alpha * beta * P_sw,d

            #invert this and create interpolant on a very fine irregular grid

            D_interpolant_irreg = sp_intpl.LinearNDInterpolator(points = np.reshape(np.stack((log_mdot_d_mesh, zeta_mesh), axis = -1), (len(D_d)*len(zeta), 2)), values = np.ndarray.flatten(D_d_mesh))

            #evaluate this on a regular grid of mdot and zeta


            return D_interpolant_irreg   #, H_interpolant_reg


    def get_height_and_depth_dimensionless(self, log_mdot_d):

        depth_d = self.D_interpolant_irreg(log_mdot_d, zeta)

        height_d = np.sqrt(depth_d**2 - self.zeta)

        return height_d, depth_d


    def get_height_and_depth(self, position, velocity):

        alpha = self.get_alpha(position, velocity)

        beta = self.get_beta(position, velocity)

        zeta = self.get_zeta(position, velocity)

        u = np.sqrt(np.sum(velocity**2, axis = 1))

        log_mdot_d = np.log(self.mdot * self.c_s**2 / (np.pi * u * alpha * beta * self.get_PswD(position, velocity))) #P_sw, D

        depth = self.D_interpolant_irreg(log_mdot_d, zeta) * alpha

        height = np.sqrt(depth**2 / alpha**2 - zeta) * beta

        return height, depth


    def get_density(self, a, b, position, velocity, depth):

        alpha = self.get_alpha(position, velocity)

        beta = self.get_beta(position, velocity)

        P_swD = self.get_PswD(position, velocity)

        rho_0 = P_swD / self.c_s**2 * np.exp(depth**2 / alpha**2)

        return rho_0 * np.exp(- a**2 / alpha**2 - b**2 / beta **2)


class Constant():

    def __init__(self, mdot, height, depth):

        self.mdot = mdot
        self.height = height
        self.depth = depth


    def get_height_and_depth(self, position, velocity):

        return self.height * np.ones(np.shape(position[:, 0])), self.depth * np.ones(np.shape(position[:, 0]))


    def get_density(self, a, b, position, velocity, depth):

        u = np.sqrt(np.sum(velocity**2, axis = 1))

        return self.mdot / (np.pi * self.height * self.depth * u) * np.ones(np.shape(depth))
