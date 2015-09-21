
import numpy as np
cimport numpy as cnp
cimport cython

from libc.math cimport sqrt, exp, fabs, cos, sin, tan, acos

cdef double PI = 3.1415926535897932

cdef double cot(double d):
	return 1/tan(d)

cdef extern from "complex.h":
	double cargl(double complex)

cdef double [:] EulerAngles(double [:] input):
	cdef:
		double x
		double y
		double z
		double [:] output
		double complex complex_xy

	x = input[0]
	y = input[1]
	z = input[2]
	output = np.zeros(2)

	if x*x < 10e-6 and y*y < 10e-6 and (z-1)*(z-1) < 10e-6: 	# handle the case (0,0,1)
		output[0] = 0
		output[1] = 0

	elif x*x < 10e-6 and y*y < 10e-6 and (z+1)*(z+1) < 10e-6: 	# handle the case (0,0,-1)
		output[0] = PI
		output[1] = 0

	else:														# all other cases
		output[0] = acos(z)
		complex_xy = complex(x,y)
		output[1] = cargl(complex_xy)

	return output

cdef double [:,:] R(double [:] input):
	
	cdef:
		double beta
		double gamma
		double [:,:] output
		double cb
		double sb
		double cg
		double sg

	beta = input[0]
	gamma = input[1]
	output = np.zeros((3,3))

	cb = cos(beta)
	sb = sin(beta)
	cg = cos(gamma)
	sg = sin(gamma)

	output[0,0] = cb*cg
	output[0,1] = -sg
	output[0,2] = cg*sb
	output[1,0] = cb*sg
	output[1,1] = cg
	output[1,2] = sb*sg
	output[2,0] = -sb
	output[2,1] = 0
	output[2,2] = cb

	return output


cdef class EnhancementKernel:

	cdef double D33
	cdef double D44
	cdef double t

	def __init__(self, D33, D44, t):
		self.D33 = D33
		self.D44 = D44
		self.t = t

		# add here:
		#	- check if lookup table file exists
		#	- load if available, else create
		#	- define orientations. use 162 directions

	def Create(self):
		pass

	cdef double GetMaximumValue(self):
		pass

	def EstimateKernelSize(self):
		pass

	def k2(self, double [:] x, double [:] y, double [:] r, double [:] v):

		cdef:
			double [:] a
			double [:,:] transm
			double [:] arg1
			double [:] arg2p
			double [:] arg2
			double [:] c
			double kernelval

		a = np.subtract(x,y)
		transm = np.transpose(R(EulerAngles(v)))
		arg1 = np.dot(transm,a)
		arg2p = np.dot(transm,r)
		arg2 = EulerAngles(arg2p)

		c = self.coordinateMap(arg1[0], arg1[1], arg1[2], arg2[0], arg2[1])
		kernelval = self.kernel(c)

		return kernelval

	cdef double [:] coordinateMap(self, double x, double y, double z, double beta, double gamma):

		cdef:
			double [:] c
			double q
			double cg
			double cotq2
		c = np.zeros(6)

		if beta == 0:
			c[0] = x
			c[1] = y
			c[2] = z
			c[3] = c[4] = c[5] = 0

		else:
			q = fabs(beta)
			cg = cos(gamma)
			sg = sin(gamma)
			cotq2 = cot(q/2)

			c[0] = -0.5*z*beta*cg + x*(1-(beta*beta*cg*cg*(1 - 0.5*q*cotq2))/(q*q)) - (y*beta*beta*cg*(1-0.5*q*cotq2)*sg)/(q*q)
			c[1] = -0.5*z*beta*sg - (x*beta*beta*cg*(1-0.5*q*cotq2)*sg)/(q*q) + y*(1-(beta*beta*(1-0.5*q*cotq2)*sg*sg)/(q*q))
			c[2] = 0.5*x*beta*cg + 0.5*y*beta*sg + z*(1+((1-0.5*q*cotq2)*(-beta*beta*cg*cg - beta*beta*sg*sg))/(q*q))
			c[3] = beta * (-sg)
			c[4] = beta * cg
			c[5] = 0

		return c

	cdef double kernel(self, double [:] c):
		return 1/(8*sqrt(2))*sqrt(PI)*self.t*sqrt(self.t*self.D33)*sqrt(self.D33*self.D44) * 1/(16*PI*PI*self.D33*self.D33*self.D44*self.D44*self.t*self.t*self.t*self.t) * exp(-sqrt( (c[0]*c[0] + c[1]*c[1])/(self.D33*self.D44) + (c[2]*c[2]/self.D33 + (c[3]*c[3]+c[4]*c[4])/self.D44)*(c[2]*c[2]/self.D33 + (c[3]*c[3]+c[4]*c[4])/self.D44) + c[5]*c[5]/self.D44)/(4*self.t));








