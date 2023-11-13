import numpy as np

class ContinuousRV():
    """
    Z = ContinuousRV(f, a=0, b=1, res=0.001, epislon=0.001)

    f: function/callable such that for y=f(x) , y must be a float or int for each a <= x <= b 

    This class allows us to create a random variable Z by specifying a probability density
    function f and support [a,b]. If the pdf does not integrate to one over its support, it will be automatically normalised.

    The pdf is approximated by an optimal and minimal set of points (x_i, y_i). 
    Points are chosen by solving an optimisation problem related to the integral of the 2nd derivative of f(x).
    Areas of greater curvature are approximated by closer sets of points. For more details see github.com/chriskward.
    
    Once the class is instantiated, the function f is discarded. All probability calculations and the generation
    of samples use this minimal set of points and a collection of interpolation routines.


    """
    




    def __init__(self, f, a=0, b=1, res=0.001, epsilon=0.001):
        """
        f: probability density function (pdf) accepting a single float:x and returning a float:y

        [a,b] int or float specifying the upper and lower bounds of the domain of f

        res=0.001
        epsilon=0.001

        These parameters relate to the optimisation routine. Smaller values for greater accuracy
        at greater computational cost
        """


        if not hasattr(f, '__call__') : raise TypeError("Probability Density Function must be of type 'function'")

        # the upper and lower bounds of the support
        
        self.a = a
        self.b = b

        # the resolution and epsilon parameter for the optimisation
        # routine that approximates the smooth function 'f' with a
        # series of points

        self.res = res
        self.epsilon = epsilon

        # approximate the prob. density function with an array of 
        # optimally spaced points (x,y)

        self.pdf_x , self.pdf_y = self._rectify(f, self.a, self.b, self.res, self.epsilon)

        #normalise to ensure pdf integrates to one

        norm_constant = self._auc(self.pdf_x, self.pdf_y, self.a, self.b)
        self.pdf_y *= 1/norm_constant

        # approximate the cumulative distribution function F(x) with
        # a series of optimally spaced points (x,y)

        self.cdf_x = np.linspace(a+self.res,b, int((b-a)/res))
        
        cdf = []
        for x in self.cdf_x:
            area = self._auc(self.pdf_x, self.pdf_y, self.a, x)  # integrate the pdf from 'a' to each value of x in the supprt, eg a+1*res , a+2*res , ....
            cdf.append(area)

        self.cdf_y = np.array(cdf)



    def D(self,x):
        """
        .D(x) -> float

        Returns the value y of the probability density function at f(x)
        """


        return self._interpolate(self.pdf_x,self.pdf_y,x)


    def F(self,x):
        """
        .F(x) -> float

        The cumulative distribution function. Returns the probability P(X<=x)
        """



        if x>=self.b : x=self.b
        return self._interpolate(self.cdf_x,self.cdf_y,x)


    def P(self, a, b):
        """
        .P(a,b) -> float

        returns the probability P(a<= X <= b)
        """


        if a<=self.a : a=self.a
        if b>=self.b : b=self.b
        f_b = self._interpolate(self.cdf_x,self.cdf_y,b)
        f_a = self._interpolate(self.cdf_x,self.cdf_y,a)
        return f_b - f_a


    def sample(self, n):
        """
        .sample( no_of_samples ) -> nd.array.shape(no_of_samples,)

        samples from the random variable (by inversion)
        """


        u = np.random.rand(n)
        out = []

        for i in u:
            out.append( self._interpolate(self.cdf_y, self.cdf_x, i) )

        return np.array(out)



    def _interpolate(self, xarray, yarray, x):

        xarray = xarray.copy()
        yarray = yarray.copy()

        if x < xarray.min() or x > xarray.max() : return 0
        if (xarray == x).any(): return yarray[ (xarray == x) ][0]

        max_arg = (xarray > x).argmax()
        min_arg = max_arg-1

        x_percent_dist =  (x - xarray[min_arg]) / (xarray[max_arg] - xarray[min_arg])  

        return yarray[min_arg] + x_percent_dist * (yarray[max_arg] - yarray[min_arg])



    def _rectify(self, f, a, b, res , epsilon):


        x = np.linspace(a,b, int((b-a)/res))
        y = f(x)

        abs_second_derv = abs( np.gradient( np.gradient(y,x) , x) )
        auc_second_derv = abs_second_derv * res
        cumulative_auc = np.cumsum(auc_second_derv)

        x_out , y_out = list([ x[0] ]) , list([ y[0] ])


        while cumulative_auc.max() > 0 :
            i = (cumulative_auc > epsilon).argmax()
            if i != 0:
                x_out.append( x[i] )
                y_out.append( y[i] )
            cumulative_auc -= epsilon

        x_out.append(x[-1])
        y_out.append(y[-1])


        return np.array(x_out), np.array(y_out)



    def _auc(self, xarray,yarray,a, b):

        xarray = xarray.copy()
        yarray = yarray.copy()

        if a == None or a<xarray.min() : a = xarray.min()
        if b == None or b>xarray.max() : b = xarray.max()

        lower_cut = (xarray<=a).argmin()-1
        upper_cut = (xarray<b).argmin()+1

        xarray = xarray[lower_cut:upper_cut]
        yarray = yarray[lower_cut:upper_cut]

        #percentage deviance from lower_cut and a - scale the first element of area
        #likewise for upper_cut ...

        lower_y_scale = xarray[1]-a / xarray[1]-xarray[0]
        upper_y_scale = b-xarray[-2] / xarray[-1]-xarray[-2]

        xarray[0]  = a
        xarray[-1] = b

        yarray[0] =  yarray[1]  - ( (yarray[1]-yarray[0])   *lower_y_scale)
        yarray[-1] = yarray[-1] - ( (yarray[-1]-yarray[-2]) *upper_y_scale)

        dx = np.append( np.diff(xarray),[0] )
        dy = np.append( np.diff(yarray),[0] )

        area = (dx*yarray) + (dx*dy*0.5)

        return np.sum(area)


class DiscreteRV():

    def __init__(self, f, support=[0,1,2,3]):

        if not hasattr(support, '__iter__') : raise TypeError('Support must be an iterable type')
        if not hasattr(f, '__call__') : raise TypeError("Probability Density Function must be of type 'function'")

        # create arrays for the prob. mass function

        self.pmf_x = np.array(support)

        # this should be adjusted to take dict input as
        # well as a function. eg, it would be good to be
        # able to specify a pdf by a dict {0:0.2, 1:0.6, 2:0.2}

        probs = []
        for i in self.pmf_x:
            probs.append(f(i))

        self.pmf_y = np.array(probs)
        norm_constant = self.pmf_y.sum()
        self.pmf_y *= 1/norm_constant

        # create arrays for the cumulative distribution function

        self.cdf_x = self.pmf_x
        self.cdf_y = np.cumsum(self.pmf_y)

    def D(self,x):
        if x in self.pmf_x: return self.pmf_y[ self.pmf_x == x ][0]
        else: return 0

    def F(self,x):
        if x>=self.cdf_x.max() : x=self.cdf_x.max()
        index = (self.cdf_x <= x).argmin()-1
        return self.cdf_y[index]

    def P(self,a,b):
        if a<self.cdf_x.min() : a=self.cdf_x.min()
        if b>self.cdf_x.max() : b=self.cdf_x.max()

        upper_index = (self.cdf_x<= a).argmin()-1
        lower_index = (self.cdf_x> b).argmax()-1

        return self.cdf_y[upper_index] - self.cdf_y[lower_index]

    def sample(self,n):

        u = np.random.rand(n)
        out = []

        for i in u:
            index = (self.cdf_y <= i).argmin()-1
            out.append(self.cdf_x[index])

        return np.array(out)



def normal(mean=0,var=1,a=-10,b=10):
    """
    normal(mean = 0 , var = 1, a = -10 , b = 10) -> ContinuousRV


    returns an instance of ContinuousRV with the normal distribution parameterised as specified
    """

    f = lambda x: ( np.sqrt(2*np.pi*var) )**-1 * np.exp( -1/(2*var) * (x-mean) **2 )

    return ContinuousRV(f,a,b)



def exponential(theta = 1,b=10):
    """
    exponential(theta = 1, b = 10) -> ContinuousRV


    returns an instance of ContinuousRV with the exponential distribution parameterised as specified.
    theta is the rate parameter (or inverse of the scale parameter)
    """

    f = lambda x: theta*np.exp(-theta*x)

    return ContinuousRV(f,a=0,b=b)






