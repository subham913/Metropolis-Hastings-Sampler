from matplotlib import pyplot as plt
import numpy as np
import copy

sig_2 = [0.01,1,100]
# sig_2 = [0.01]
true_mu = np.array([4,4])
true_cov = np.array([[1,0.8],[0.8,1]])
sample_size = 10000
orig_samples = np.random.multivariate_normal(true_mu, true_cov, sample_size)

def gaussian_ratio(x1,x2,mu, cov):
    return np.exp(-0.5*np.dot(np.dot((x1 - mu).T,np.linalg.inv(cov)),(x1 - mu))+0.5*np.dot(np.dot((x2 - mu).T,np.linalg.inv(cov)),(x2 - mu)))

def gaussian(x, mu, cov):
    return np.exp(-0.5*np.dot(np.dot((x - mu).T,np.linalg.inv(cov)),(x - mu)))/(2*np.pi*np.sqrt(np.linalg.det(cov)))

def p(x,mu,cov):
    from scipy.stats import multivariate_normal
    return multivariate_normal.pdf(x,mean=mu,cov=cov)

def MH_sampling(sig2):
    samples = []
    temp_real = np.asarray(orig_samples)
    x1_real = temp_real[:,0]
    x2_real = temp_real[:,1]
    i=1

    # plt.show()
    plt.subplot(2,2,i)
    x, y = np.mgrid[-2:8:.01, -2:8:.01]
    pos = np.dstack((x, y))
    cs=plt.contour(x, y, p(pos,true_mu,true_cov), levels=[0.03], colors="blue", linewidths=3)
    h,_ = cs.legend_elements()
    plt.legend([h[0]], ['true distribution'])

    plt.scatter(x1_real,x2_real,c='red',label="Samples")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("10000 samples from true distribution")
    i+=1
    cov = np.array([[sig2,0],[0,sig2]])
    mean = np.array([2,2])
    z_prev = np.random.multivariate_normal(mean, cov, 1)
    z_prev=z_prev.reshape(2,)
    # print(z_prev.shape)
    # print(z_prev)
    trial = 0
    while(len(samples)<sample_size):
        zCand = np.random.multivariate_normal(z_prev, cov, 1)
        trial+=1
        # print(zCand.shape)
        zCand = zCand.reshape(2,)
        u = np.random.uniform(0,1,1)
        criteria = min(1,gaussian_ratio(zCand,z_prev,true_mu,true_cov))
        if u<criteria:
            z_prev = copy.deepcopy(zCand)
            samples.append(zCand)
            if (len(samples)==int(sample_size/100) or len(samples)==int(sample_size/10) or len(samples)==sample_size):
                temp = np.asarray(samples)
                x1 = temp[:,0]
                x2 = temp[:,1]
                sample_cov=np.cov(temp.T)
                sample_mu=np.mean(temp,axis=0)
                x, y = np.mgrid[-2:8:.01, -2:8:.01]
                pos = np.dstack((x, y))
                plt.subplot(2,2,i)
                s=p(pos,true_mu,true_cov)
                # print(s.shape)
                cs1=plt.contour(x, y, p(pos,true_mu,true_cov), levels=[0.03], colors="blue", linewidths=3)
                cs2=plt.contour(x, y, p(pos,sample_mu, sample_cov), levels=[0.03], colors="green", linewidths=3)
                h1,_ = cs1.legend_elements()
                h2,_ = cs2.legend_elements()
                plt.legend([h1[0], h2[0]], ['true distribution', 'proposal distribution'])
                plt.scatter(x1,x2,c='red',label="Samples")
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.title("for sample_size= "+str(len(samples))+" and sig2= "+str(sig2))
                i+=1
    plt.show()
    plt.close()
    return np.asarray(samples),trial



for i in range(len(sig_2)):
    samples,trial=MH_sampling(sig_2[i])
    rejected_samples = trial-sample_size
    rejection_rate = rejected_samples/trial
    print("-----rejection rate for sig2 = "+str(sig_2[i])+" is "+str(rejection_rate)+" -----")
    # print(samples.shape)
    # print(samples.max(),samples.min())
