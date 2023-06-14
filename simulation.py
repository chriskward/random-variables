import numpy as np

class ContinuousRV():

    def __init__(self, f, a=0, b=1, res=0.001, epsilon=0.001):

        # f must be callable (and capable of handling np.array input)

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

        # approximate the cumulative distribution function F(x) with
        # a series of optimally spaced points (x,y)

        self.cdf_x = np.linspace(a+self.res,b, int((b-a)/res))
        
        cdf = []
        for x in self.cdf_x:
            area = self._auc(self.pdf_x, self.pdf_y, self.a, x)  # integrate the pdf from 'a' to each value of x in the supprt, eg a+1*res , a+2*res , ....
            cdf.append(area)

        self.cdf_y = np.array(cdf)



    def D(self,x):
        return self._interpolate(self.pdf_x,self.pdf_y,x)


    def F(self,x):
        if x>=self.b : x=self.b
        return self._interpolate(self.cdf_x,self.cdf_y,x)


    def P(self, a, b):
        if a<=self.a : a=self.a
        if b>=self.b : b=self.b
        f_b = self._interpolate(self.cdf_x,self.cdf_y,b)
        f_a = self._interpolate(self.cdf_x,self.cdf_y,a)
        return f_b - f_a


    def sample(self, n):
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
