import numpy as np

def _gaussian_fn(x, mu, sigma):
    return np.exp(-(x - mu) * (x - mu) / (2 * sigma*sigma))

def _planck_fn(l, T):
    return 0.5*10**18 * l**-5. * ( 1/(np.exp(4000/l)-1 ))
    
def _noise_generator(size):
    """
    amplitudes = abs(np.random.normal(0.0, scale=1., size=size))
    noise = np.random.uniform(amplitudes, -amplitudes, size=size)
    """
    noise = np.random.normal(0.0, scale=1., size=size)
    return noise

def gen_4types(batch_size, length, labels=None):
    labels_active = False
    unit_len = length/60
    labels_dict = {0: [100,1000], 
                   1: [100,1000],
                   2: [2500,3400],
                   3: [2500,3400]}
    if labels==None:
        labels = []
    else:
        labels_active = True
        if type(labels)==int:
            labels = [labels]
        if len(labels)!=batch_size:
            raise ValueError('Each value needs a mode (label)!')
    y = []
    x = np.arange(1,length+1)
    for i in range(batch_size):
        if not labels_active:
            labels.append(np.random.randint(0,4))
        noise = _noise_generator(size=length)

        variance_limits = (0.3*length/1000,5*length/1000)
        amplitude_limits = (5,8)

        variance = np.random.uniform(variance_limits[0],variance_limits[1],size=1)
        amplitude = np.random.uniform(amplitude_limits[0], amplitude_limits[1])
        
        if labels[i]==1 or labels[i]==3:
            amplitude = -amplitude

        gaussian = amplitude*_gaussian_fn(np.arange(int(6*variance)), 
                                          np.random.uniform(0, int(6*variance)), 
                                          variance) 

        center_limits = labels_dict[labels[i]]
        center = int(np.random.uniform(center_limits[0], center_limits[1], size=1))

        left = center - int(3*variance)
        right = left + int(6*variance)
        noise[left:right] += gaussian

        y.append(noise.astype(np.float32))
    return [x for i in range(batch_size)], np.array(y), np.array(labels)


def gen_spectrum_2lines(batch_size, length, labels=None, norm=False):
    labels_active = False
    unit_len = length/60
    labels_dict = {0: int(unit_len*2), 
                   1: int(unit_len*8),
                   2: int(unit_len*14),
                   3: int(unit_len*20)}
    if labels==None:
        labels = []
    else:
        labels_active = True
        if type(labels)==int:
            labels = [labels]
        if len(labels)!=batch_size:
            raise ValueError('Each value needs a mode (label)!')
    y = []
    x = np.arange(1,length+1)
    for i in range(batch_size):
        if not labels_active:
            labels.append(np.random.randint(0,4))
        noise = 0.1*_noise_generator(size=length)

        variance_limits = (0.3*length/1000,5*length/1000)
        amplitude_limits = (.4,.7)

        gaus_number = 4
        variances = np.random.uniform(variance_limits[0],variance_limits[1],size=gaus_number)
        amplitudes = []
        for i_gaus in range(gaus_number):
            up_or_down = np.random.randint(0,2)
            if up_or_down:
                amplitude_limits = (-amplitude_limits[1], -amplitude_limits[0])
            if i_gaus==1: #fix the amplitude of second line relative to the first
                if amplitudes[0]>=0:
                    amplitudes.append(amplitudes[0]+.2)
                else:
                    amplitudes.append(amplitudes[0]-.2)
            else:
                amplitudes.append(np.random.uniform(amplitude_limits[0],amplitude_limits[1],size=1))

        gaussians = [amplitudes[j]*_gaussian_fn(np.arange(int(6*variances[j])), 
                                               np.random.uniform(0, int(6*variances[j])), 
                                               variances[j]) for j in range(gaus_number)]

        center1 = int(np.random.uniform(3*variances[0], 
                                        length-labels_dict[labels[i]]-3*variances[1], size=1))
        center2 = center1 + labels_dict[labels[i]]
        center3 = np.random.randint(np.ceil(3*variances[2]),np.floor(length-3*variances[2])+1)
        center4 = np.random.randint(np.ceil(3*variances[3]),np.floor(length-3*variances[3])+1)
        #center5 = np.random.randint(np.ceil(3*variances[4]),np.floor(length-3*variances[4])+1)
        #center6 = np.random.randint(np.ceil(3*variances[5]),np.floor(length-3*variances[5])+1)
        centers = np.array([center1, center2, center3, center4])
        #centers = np.array([center1, center2, center3, center4, center5, center6])

        for k in range(gaus_number):
            left = centers[k] - int(3*variances[k])
            right = left + int(6*variances[k])
            noise[left:right] += gaussians[k]

        if norm:
            noise = np.tanh(noise)
        y.append(noise.astype(np.float32))
    return [x for i in range(batch_size)], np.array(y), np.array(labels)
