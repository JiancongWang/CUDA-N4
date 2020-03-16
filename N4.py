#%% This is a 3D python version of the N4 algorithm. Used to understand the algorthm
# Written by Jiancong Wang, PICSL Lab, Upenn. 
import numpy as np
from numpy.fft import fft, ifft, ifftshift, rfft, irfft
import nibabel as nib
from scipy.ndimage import zoom as zoom
import matplotlib.pyplot as plt

def sharpenImage(im_before_sharpen, 
                 mask, 
                 numberOfHistogramBins = 200, 
                 WienerFilterNoise = 0.01, 
                 BiasFieldFullWidthAtHalfMaximium = 1.5):
    binMin, binMax = im_before_sharpen[mask!=0].min(), im_before_sharpen[mask!=0].max()
    histogramSlope = (binMax - binMin)/(numberOfHistogramBins-1)
    H = np.zeros(numberOfHistogramBins)
    
    # Build a fraction histogram
    for pixel, mask_value in zip(im_before_sharpen.flatten(), mask.flatten()):
        if mask_value!=0:
            cidx = ( pixel - binMin ) / histogramSlope;
            idx = int(np.floor( cidx ))
            offset = cidx - idx
            
            if offset == 0.0:
                H[idx] += 1.0
            elif( idx < numberOfHistogramBins - 1 ):
                H[idx] += 1.0 - offset
                H[idx+1] += offset
        
    
    exponent = np.ceil( np.log( numberOfHistogramBins ) /
              np.log( 2.0 ) ) + 1;
    
    paddedHistogramSize = (int)( 2.0 ** exponent + 0.5 )
    
    histogramOffset = (int)( 0.5 * ( paddedHistogramSize - numberOfHistogramBins ) )
    
    V = np.zeros(paddedHistogramSize)
    
    V[histogramOffset:histogramOffset+numberOfHistogramBins] = H
    
    Vf = fft(V)
    
    # Make the gaussian kernel
    scaledFWHM = BiasFieldFullWidthAtHalfMaximium / histogramSlope
    expFactor = 4.0 * np.log( 2.0 ) / ( scaledFWHM * scaledFWHM )
    scaleFactor = 2.0 * np.sqrt( np.log( 2.0 ) / np.pi ) / scaledFWHM
    
    F = np.zeros(paddedHistogramSize)
    F[0] = scaleFactor
    
    halfSize = (int)(0.5 * paddedHistogramSize)
    
    for i in range(1, halfSize+1):
        F[i] = F[paddedHistogramSize - i] = scaleFactor * np.exp(-i*i*expFactor)
        
    if (paddedHistogramSize%2==0):
        F[halfSize] = scaleFactor * np.exp(-0.25* paddedHistogramSize*paddedHistogramSize*expFactor)
        
    Ff = fft(F)
    
    c = np.conj(Ff)
    
    Gf = c / (c*Ff + WienerFilterNoise)
    
    Uf = Vf * np.real(Gf)
    
    U = ifft(Uf)
    
    # Keep the the real part only and them to >=0
    U = np.real(U)
    U[U<0] = 0
    
    numerator = np.zeros(paddedHistogramSize, dtype = np.complex64)
    
    for i in range(paddedHistogramSize):
        numerator[i] = (binMin + (i - histogramOffset)* histogramSlope)* U[i]
        
#    numerator_before_smoothing = numerator
        
    numerator_f = fft(numerator)
    numerator_f = numerator_f * Ff
    numerator = ifft(numerator_f)
    
    denominator = U
    denominator_f = fft(U)
    denominator_f = denominator_f * Ff
    denominator = ifft(denominator_f)
    
    E = np.zeros(paddedHistogramSize)
    for i in range(paddedHistogramSize):
        nu = numerator[i].real
        de = denominator[i].real
        
        if de!=0:
            E[i] = nu/de
        else:
            E[i] = 0.
    
    E = E[histogramOffset:histogramOffset+numberOfHistogramBins]
    
    im_after_sharpen = np.zeros_like(im_before_sharpen).flatten()
    
    for i, (pixel, mask_value) in enumerate(zip(im_before_sharpen.flatten(), mask.flatten())):
        if mask_value!=0:
            cidx = ( pixel - binMin ) / histogramSlope;
            idx = int(np.floor( cidx ))
            
            if (idx < numberOfHistogramBins - 1):
                im_after_sharpen[i] = E[idx] + ( E[idx + 1] - E[idx] ) * ( cidx -  idx  );
            else:
                im_after_sharpen[i] = E[-1]
    
    im_after_sharpen = np.reshape(im_after_sharpen, im_before_sharpen.shape)
    
#    return im_after_sharpen, H, E, F
    return im_after_sharpen


def CalculateConvergenceMeasurement( field1, field2, mask ):
    field_diff = field1 - field2 
    pixel = np.exp(field_diff)[mask!=0]
    mu = np.mean(pixel)
    sigma = np.std(pixel)
    return sigma/mu


# This function upsample the lattice grid according to the equation defined in the paper
# Scattered Data Interpolation with Multilevel B-Splines, section 4.2. 
# This function always upsample a cubic Bspline lattice of size (m+3)*(n+3)*(l+3) to (2m+3)*(2n+3)*(2l+3)
def UpsampleLattice3D(lattice):
    # This assumes the array index starts from -1. The offseting is done expliciting by adding 1 to every index to this.
    bw_0 = np.array([1/8, 6/8, 1/8]) # For coefficient i,j,k
    bw_1 = np.array([0, 1/2, 1/2]) # For coefficient i+1/j+1/k+1
    bw = [bw_0, bw_1]
    
    # Get the n,m,l
    n, m, l = lattice.shape
    n-=3
    m-=3
    l-=3
    
    # Initialize the upsampled lattice
    lattice_upsample = np.zeros((2*n+3, 2*m+3, 2*l+3))
    for i in range(-1, n+2):
        for j in range(-1, m+2):
            for k in range(-1, l+2):
                # Each of this involves points i/j/k: -1, 0, 1.         
                # This gets lattice pieces to work with
                X, Y, Z = np.meshgrid( range(i-1, i+2), range(j-1, j+2), range(k-1, k+2), indexing = 'ij')
                X = np.clip(X, -1, n+1) # Simply use a constant boundary
                Y = np.clip(Y, -1, n+1)
                Z = np.clip(Z, -1, n+1)
                lattice_piece = lattice[X+1,Y+1,Z+1] # +1 here since the above computation assume the array starting index to be -1.
                
                # (2i: 2i+1, 2j:2j+1, 2k:2k+1)
                for xx, idx in enumerate(range(2*i, 2*i+2)):
                    for yy, idy in enumerate(range(2*j, 2*j+2)):
                        for zz, idz in enumerate(range(2*k, 2*k+2)):
                            if (idx>=-1 and idx<=2*n+1) and (idy>=-1 and idy<=2*n+1) and (idz>=-1 and idz<=2*n+1):
                                bw_outer = bw[xx][:, None, None] * bw[yy][None, :, None] * bw[zz][None, None, :] 
                                lattice_upsample[idx + 1, idy + 1, idz + 1] = (bw_outer* lattice_piece).sum() 
                                # Each upsampled value are weighted average from the 27 number
    return lattice_upsample

#a = np.arange(1, 10)
#lo = a[:, None, None] * a[None, :, None] * a[None, None, :] 
#hi = UpsampleGrid3D(lo)

#  This function calculates the cubic B spline weight given position t
# Scattered Data Interpolation with Multilevel B-Splines, 3.1

def cubicBspline(t):
    t2 = t**2
    t3 = t**3
    B0 = (1-t)**3/6
    B1 = (3*t3 - 6*t2 + 4)/6
    B2 = (-3*t3 + 3*t2 + 3*t + 1)/6
    B3 = t3/6
    return np.array([B0, B1, B2, B3])
    
# This only needs to take care of one level of fitting as the original N4 
# implementation only uses 1 level of fitting each time.
# numberOfControlPoints is a 3 tuple.
def FitBspline3D(field, mask, numberOfControlPoints, eps = 1e-3):
    numberOfControlPoints = np.array(numberOfControlPoints)
    size = np.array(field.shape)
    X, Y, Z = size
    
    numberOfSpans = numberOfControlPoints - 3
    spans = size / numberOfSpans
    
    # Initialize for the control points: the wc^2 * phi_c and wc^2.
    # Need to accumulate these 2 terms to calculate the B-spline values
    wc2_phic = np.zeros(numberOfControlPoints)
    wc2 = np.zeros(numberOfControlPoints)
    
    # Loop through all pixels - Calculate wc and sum(wc^2) for each point
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                # Determine which B spline control points this point contribute to
                # This gives you the index of control point immediately left to current pixel position.
                # This also happens to be the lattice index for a cubic B-spline since the lattice index starts with -1.
                # Here 0.5 is assuming the position of width 1 and its value lies in the middle.
                if mask[i, j, k]==0:
                    continue
                # The index within the control point lattice
                bx = int(np.floor((i+0.5) / spans[0]))
                by = int(np.floor((j+0.5) / spans[1]))
                bz = int(np.floor((k+0.5) / spans[2]))
                
                # The normalized local coordinates t for calculating bspline coefficients
                tx = ((i+0.5) - bx * spans[0])/spans[0]
                ty = ((j+0.5) - by * spans[1])/spans[1]
                tz = ((k+0.5) - bz * spans[2])/spans[2]
                
                # b spline coefficient
                wx = cubicBspline(tx)
                wy = cubicBspline(ty)
                wz = cubicBspline(tz)
                
                # outer product. Calculate the wc
                wc = wx[:, None, None]*wy[None, :, None]*wz[None, None, :]
                wc_sum = (wc**2).sum()
                
                # calculate the phi_c for each control points
                phi_c = field[i,j,k] * wc / wc_sum
                
                # Each control points, Accumulate wc^2 * phi_C and wc^2
                wc2_phic[bx:bx+4, by:by+4, bz:bz+4] += (wc**2) * phi_c
                wc2[bx:bx+4, by:by+4, bz:bz+4] += (wc**2)
                
    # Final fitting result, sum(wc^2 * phi_C) / sum(wc^2)
    lattice = wc2_phic/(wc2 + eps)
    return lattice

# This function given a fitted Bspline lattice and an image shape, return the interpolation result
# on the image.
def EvaluateBspline3D(lattice, numberOfControlPoints, size):
    numberOfControlPoints = np.array(numberOfControlPoints)
    X, Y, Z = size
    numberOfSpans = numberOfControlPoints - 3
    spans = size / numberOfSpans
    
    fitted = np.zeros(size)
    
    # Loop through all pixels - Calculate wc and sum(wc^2) for each point
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                # The index within the control point lattice
                bx = int(np.floor((i+0.5) / spans[0]))
                by = int(np.floor((j+0.5) / spans[1]))
                bz = int(np.floor((k+0.5) / spans[2]))
                
                # The normalized local coordinates t for calculating bspline coefficients
                tx = ((i+0.5) - bx * spans[0])/spans[0]
                ty = ((j+0.5) - by * spans[1])/spans[1]
                tz = ((k+0.5) - bz * spans[2])/spans[2]
                
                # b spline coefficient
                wx = cubicBspline(tx)
                wy = cubicBspline(ty)
                wz = cubicBspline(tz)
                
                # outer product. Calculate the wc
                wc = wx[:, None, None]*wy[None, :, None]*wz[None, None, :]
                
                fitted[i,j,k] = (lattice[bx:bx+4, by:by+4, bz:bz+4]*wc).sum()
    
    return fitted
                
# This is the master function of the N4 algorithm.
def N4(im, mask, low_value = 10, 
       maximumNumberOfLevels = 1,
       maximumNumberOfIterations = [50],
       ConvergenceThreshold = 0.001):
    im, mask = im.copy(), mask.copy()
    
    # Remove the value below low_value (default 10) on the image. This should be adjusted 
    # to remove tails from the image histogram.
    mask[im<=low_value] = 0
    im[im<=low_value] = 0
    
    im_log = np.log(im) # Log the image
    im_log[im<=low_value]
    
    logUncorrectedImage = im_log.copy() # Make a duplicate of the log image
    logBiasField = np.zeros_like(im)
    lattice = np.zeros((4,4,4)) # Initial lattice. 2**0+3. Each time the lattice get upsampled the number of points double
    
    for currentLevel in range(maximumNumberOfLevels):
        numberOfControlPoints = lattice.shape
        
        CurrentConvergenceMeasurement = 10000000000.0
        elapsedIterations = 0
        
        maxIter = maximumNumberOfIterations[currentLevel]
        while (CurrentConvergenceMeasurement > ConvergenceThreshold and 
               elapsedIterations<maxIter):
            
            print("Level %d, iter %d" % (currentLevel, elapsedIterations))
            elapsedIterations+=1
            
            logSharpenedImage = sharpenImage(logUncorrectedImage, 
                 mask, 
                 numberOfHistogramBins = 200, 
                 WienerFilterNoise = 0.01, 
                 BiasFieldFullWidthAtHalfMaximium = 0.15)
            
            residualBiasField = logUncorrectedImage - logSharpenedImage
            
            # Fit a new Bspline lattice to the current residual field
            lattice_residual = FitBspline3D(residualBiasField, mask, numberOfControlPoints)
            lattice += lattice_residual
            newLogBiasField = EvaluateBspline3D(lattice, numberOfControlPoints, im.shape)
            CurrentConvergenceMeasurement = CalculateConvergenceMeasurement( logBiasField, newLogBiasField, mask )
            
            logBiasField = newLogBiasField # Update 
            logUncorrectedImage = im_log - logBiasField
        
        # Upsample the lattice
        if (currentLevel!=maximumNumberOfLevels-1):
            lattice = UpsampleLattice3D(lattice)
        
    # Final normalization
    logBiasField = EvaluateBspline3D(lattice, lattice.shape, im.shape)
    im_normalized = im / np.exp( logBiasField )
    
    return im_normalized, lattice


#%% Test the N4 algorithm
im = nib.load("./testImage/shrink.nii.gz")
im_np = im.get_fdata()

mask = nib.load("./testImage/shrink_mask.nii.gz")
mask_np = mask.get_fdata()
im_normalized, lattice = N4(im_np, mask_np, low_value = 10, 
                            maximumNumberOfLevels = 4, 
                            maximumNumberOfIterations = [50, 50, 50, 50], 
                            ConvergenceThreshold = 0.001)

nib.save(nib.Nifti1Image(im_normalized, im.affine), "./testImage/shrink_normalized.nii.gz")
    
