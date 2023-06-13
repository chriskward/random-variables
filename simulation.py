class ContinuousRV():

	def __init__(self, f, a=0, b=1, res=0.001):

		self.a = a
		self.b = b
		self.res = res
		pass

		#rectify the probability density function

		self.pdf_x , self.pdf_y = _rectify(f, self.a, self.b, self.res)

		#form the cumulative distribution function

		self.cdf_x = np.linspace(a,b, int((b-a)/res))
		
		cdf = []
		for x in z:
			area = _auc(self.pdf_x, self.pdf_y, self.a, x)
			cdf.append(area)

		self.cdf_y = np.array(cdf)

		"""

		methods:

		.sample(n) -> n samples
		.D(x) -> value of pdf at density x in the support
		.F(x) -> valye of cdf at x in the support
		.P(a,b) -> probability that a<x<b

		"""

	def D(self,x):
		return _interpolate(self.pdf_x,self.pdf_y,x)


	def F(self,x):
		return _interpolate(self.cdf_x,self.cdf_y,x)


	def P(self, a, b):
		f_b = _interpolate(self.cdf_x,self.cdf_y,b)
		f_a = _interpolate(self.cdf_x,self.cdf_y,a)
		return f_b - f_a


	def sample(self, n):
		u = np.random.rand(n)
		out = []

		for i in u:
			out.append( self.cdf_y, self.cdf_x, i )

		return np.array(out)



	def _interpolate(self, xarray, yarray, x):

		if x < xarray.min() or x > xarray.min() : return 0
		if (xarray == x).any(): return yarray[ (xarray == x) ][0]

		max_arg = (xarray > x).argmax()
		min_arg = max_arg-1

		x_percent_dist =  (x - xarray[min_arg]) / (xarray[max_arg] - xarray[min_arg])  

		return yarray[min_arg] + x_percent_dist * (yarray[max_arg] - yarray[min_arg])



	def _rectify(self, f, a, b, res):

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



	def _auc(xarray,yarray,a=None, b=None):

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
