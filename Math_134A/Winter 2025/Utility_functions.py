import math
from scipy.optimize import fsolve
import numpy_financial as npf


############################################################################################################


# Week one
# Computing the present value of a cash flow with interest rate r.
def PV(r, cf, continuous_compounding=False, dt=None):
    """cf: The cash flow. It can be a list of cash flows or a single cash flow
    r: The discount rate
    continuous_compounding: If this is set to True, then we will use continuous
    compounding. Otherwise, we will use discrete compounding
    dt: The time at which the cash flow is received. It can be a list of times
    or a single time. If it is a list, then it should have the same length as cf
    """
    if dt is None:
        dt = range(len(cf))
    pv = 0
    if not continuous_compounding:
        for k in range(0, len(cf)):
            pv += cf[k] / (1 + r) ** dt[k]
    else:
        for k in range(0, len(cf)):
            pv += cf[k] * math.exp(-r * dt[k])
    return pv


def FV(r, cf, continuous_compounding=False, dt=None, T=None):
    """
    This will return the future value of a cash flow at time T.
    If T is not specified, then we take T as the time that we receive the last cash flow
    cf: The cash flow. It can be a list of cash flows or a single cash flow
    r: The discount rate
    continuous_compounding: If this is set to True, then we will use continuous
    compounding. Otherwise, we will use discrete compounding
    dt: The time at which the cash flow is received. It can be a list of times
    or a single time. If it is a list, then it should have the same length as cf
    T: The time at which we want to calculate the future value. If it is not
    specified, then we will take the time of the last cash flow
    """
    if dt is None:
        dt = range(len(cf))
    if T is None:
        T = dt[-1]
    fv = 0
    if not continuous_compounding:
        for k in range(0, len(cf)):
            fv += cf[k] * (1 + r) ** (T - dt[k])
    else:
        for k in range(0, len(cf)):
            fv += cf[k] * math.exp(r * (T - dt[k]))
    return fv


# Computing the internal rate of return by using the fsolve function to find roots to a polynomial
def simple_npv(r, cf):
    """This function calculates the NPV of a cash flow given a discount rate r"""
    return sum(cf[k] / (1 + r) ** k for k in range(len(cf)))


def irr(cf):
    """This function calculates the IRR of a cash flow"""
    result = fsolve(simple_npv, x0=0.1, args=cf)
    return result[0]


# Computing the internal rate of return by using the irr function in numpy financial library
def irr2(cf):
    """This function calculates the IRR of a cash flow and to numpy"""
    return npf.irr(cf)


############################################################################################################
# This function computes the YTM of a bond mature in dt years, with face value F, price P, and coupon c paid m times every year
def YTM(F, P, dt, c = 0, m = 1):
    cf = [-P] + [c/m] * (dt * m - 1) + [F+c/m]
    return irr(cf) * m
    
# This function computes the price of a bond mature in dt years, with face value F, YTM y, and yearly coupon c paid m times every year 
def bond_price(F, y, dt, c = 0, m = 1):
    cf = [0] + [c/m] * (dt * m - 1) + [F+c/m]
    return PV(y/m, cf)

def macaulay_duration(c, y, m, n):
    """
    c is the coupon rate per period
    y is the yield per period
    m is the number of periods per year
    n is the number of periods remaining
    """
    D = (1+y)/(m*y) - (1 + y + n*(c-y))/(m*c * ((1+y)**n - 1) + m*y)
    return D