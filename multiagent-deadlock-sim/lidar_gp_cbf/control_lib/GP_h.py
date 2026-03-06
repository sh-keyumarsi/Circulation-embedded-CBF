import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import warnings

PYSIM = True
#PYSIM = False # for experiment or running via ROS

if PYSIM:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings('ignore')
'''________________________ color map ______________________________________'''
#red to green color map for safe and unsafe areas 
cdict = {'green':  ((0.0, 0.0, 0.0),   # no red at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

        'red': ((0.0, 1, 1),   # set to 0.8 so its not too bright at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }
RdGn = colors.LinearSegmentedColormap('GnRd', cdict)
'''________________________ functions ______________________________________'''

def kernel(f):      
        return lambda a, b: np.array(
        [[np.float64(f(a[i], b[j]))  for j in range(b.shape[0])]
        for i in range(a.shape[0])] )

def value_eye_fill(value,shape):
    result = np.zeros(shape)
    np.fill_diagonal(result, value)
    return result

# Calculate the relative error
def relative_error(A,x,B):
    # Check the accuracy of the solution
    # Calculate the residual (error)
    residual = np.dot(A, x) - B 
    re= np.linalg.norm(residual) / np.linalg.norm(B)
    
    if re < 1e-6:
        accurate=True
    else:
        accurate=False
    return accurate

'''________________________ GP Reg _________________________________________'''

class GP():
    def __init__(self,hypers_gp,min_d_sample,grid_size_plot,dh_dt):
        self.reset_data()
        # For Plotting
        self.init_map=True
        self.__prediction_plot = None
        self.init=False
        self.set=False
        self.grid_size=grid_size_plot
        #GP initialization
        self.hypers=hypers_gp
        self.L_2=hypers_gp[0]**-2
        self.sigma_f=self.hypers[2]
        # data set is empty
        self.N=0
        #iteration number 0 (it has not started)
        self.k=0
        #minimum sampling distance
        self.min_d_sample=min_d_sample
        self.dh_dt=dh_dt
        
    def reset_data(self):
        # X is N by n
        self.data_X = None
        # Y is N by 1
        self.data_Y = None
        self.N=None # N is the number of training data
        self.k=0
        
    def new_iter(self):
        #update iteration number
        self.k+=1
        self.data_X=None
        self.data_Y= None
        self.N=0

    def set_new_data(self, new_X, new_Y=np.array([[-2]])):
        #detected edges are assigned -1 as unsafe points
        # mean is 1 as we assume safe points are not collected so Y-1 is sampled
        # checking sampling distance requirement 
        if self.data_X is None:
            self.data_X = new_X
            self.data_Y = new_Y
            self.N=len(self.data_X)
        else:
            # checking sampling distance requirement
            dis_to_mem=np.linalg.norm(self.data_X[:,0:2]-new_X[0:2], axis=1)
            if min( dis_to_mem)> self.min_d_sample :
                self.data_X = np.append(self.data_X, new_X, axis=0)
                self.data_Y = np.append(self.data_Y, new_Y, axis=0)
                self.N=len(self.data_X)
                   
    def update_gp_computation(self,t):                  
        ktt = self.main_kernel(t, t, self.hypers)
        ktX= self.main_kernel(t, self.data_X,self.hypers)
        kXX=self.main_kernel(self.data_X, self.data_X, self.hypers)  
        #add a small value to make sure matrix is PD despite small numerical errors
        kXX=kXX+1e-10 * np.eye(kXX.shape[0])                                    
        # Use the Cholesky decomposition to factorize A
        L_cho, lower = cho_factor(kXX)
        # Solve the linear system using cho_solve
        alpha = cho_solve((L_cho, lower), ktX.T).T
        #checking if the inverse matrix is accurate
        accurate=relative_error(kXX,alpha.T, ktX.T)
        # Check the accuracy of the solution
        if not accurate:
            print("The cholesky inverse is not accurate.")
            alpha = np.linalg.solve(kXX , ktX.T).T # the expensive bit
            accurate_1=relative_error(kXX,alpha.T, ktX.T)
            if not accurate_1:
                print("The linlg.solve inverse is not accurate.")
                alpha =(np.linalg.pinv(kXX) @ ktX.T).T # the expensive bit
                
        mpost =alpha @ self.data_Y  # posterior mean
        # vpost = ktt - alpha @ ktX.T  # posterior covariance 
        # std=np.sqrt(vpost)
        # 95%confidence interval mean+-1.96std `sausage of uncertainty'
        # hgp_xq=mpost+1.96*std 
        robot_rad = 0.1
        hgp_xq=mpost+1 -robot_rad
   
        return  hgp_xq,alpha,kXX,ktt,ktX

    def get_cbf_safety_prediction(self,t,dis_f):
        """ can be only computed for one state at a time """
        #Computing dh/dx
        hgp_xq,alpha,kXX,ktt,ktX=self.update_gp_computation(t)   
        # Use the Cholesky decomposition to factorize A
        L_cho_1, lower_1 = cho_factor(kXX)
        # Solve the linear system using cho_solve       
        beta = cho_solve((L_cho_1, lower_1), self.data_Y).T
        #checking if the inverse matrix is accurate
        accurate=relative_error(kXX,beta.T, self.data_Y)
        # Check the accuracy of the solution
        if not accurate:
            # print("The cholesky inverse is not accurate.")
            beta = np.linalg.solve(kXX ,self.data_Y).T  # the expensive bit
            accurate_1=relative_error(kXX,beta.T, self.data_Y)
            if not accurate_1:
                # print("The linlg.solve inverse is not accurate.")
                beta =(np.linalg.pinv(kXX) @  self.data_Y).T # the expensive bit   
        theta = np.block([[ktX[0,i]*( self.data_X[i]-t ) ] for i in range(len(self.data_X))])
        dkdx_xq = (theta * self.L_2).T
        """ dhgpdt = dhgpdx I u """
        dmpost= np.inner(beta, dkdx_xq ) 
        # dvpost= -np.inner(2*alpha, dkdx_xq ) 
        #std= vpost^0.5=> dstd= dvpost * 1/2 * vpost^-0.5 
        # dstd=dvpost * 1/2 * vpost**-0.5 
        # dhgpdx = dmpost+1.96*dstd
        dhgpdx = dmpost 
        #print('dh/dx: ',dhgpdx)
        
        #Computing CBF linear inequality for CBF-QP
        """ -hdot <= gamaa(h)
            cvx: min 0.5 xTPx + qTx s.to: Gx <= h"""
        gp_G = -dhgpdx
        """0gp_h=gamma(hgp_xq), gamma=1"""
        # gp_h=np.exp(-1.96*np.sqrt(vpost))*hgp_xq**3
        gp_h=hgp_xq-(self.dh_dt)
        # print(gp_h,hgp_xq,vpost)
        return gp_G, gp_h, hgp_xq
    
    def main_kernel(self,a, b, hypers):
        """
        kernel BETWEEN inputs a,b  
        kernel hyperparamters:       
        ell is the characteristic length scale
        sigma_f is the signal scale
        sigma_y is the observation noise
        hypers = np.array([l_gp, sigma_f, sigma_y])
        to ensure positivity, use full float range """
        """fixed hyper params"""
        [l_gp, sigma_f, sigma_y] = hypers
        # square exponential kernel
        SE = (lambda a, b: np.exp( np.dot(a-b, a-b) / (-2.0*l_gp**2) ))
        kSE = kernel(SE)
        kSEab = kSE(a, b)
        
        # noise observation kernel
        kNOab= value_eye_fill(1,kSEab.shape)
        # main kernel is the sum of all kernels
        kab = ( sigma_f ** 2 * kSEab + sigma_y**2 * kNOab )
        return kab
                

    """................ Mapping the safety prediction..................... """
    
    def __create_mesh_grid(self, field_x, field_y):
        aa=0
        m = int( (field_x[1]+aa - field_x[0]-aa) //self.grid_size ) 
        n = int( (field_y[1]+aa - field_y[0]-aa) //self.grid_size ) 
        gx, gy = np.meshgrid(np.linspace(field_x[0]-aa, field_x[1]+aa, m), np.linspace(field_y[0]-aa, field_y[1]+aa, n))
        return gx.flatten(), gy.flatten()

    def draw_gp_whole_map_prediction(self, ax, field_x, field_y, ic, robot_pos, sensing_rad, color='r'):
        if self.init_map:
            """ initializing the mapping """
            data_point_x, data_point_y = self.__create_mesh_grid(field_x, field_y)
            r_x=data_point_x.shape[0]
            r_y=data_point_y.shape[0]
            self.t_map=np.append(np.reshape(data_point_x,(1,r_x)).T,np.reshape(data_point_y,(1,r_y)).T, axis=1)
            self.init_map=False
            
            # Assign handler, data will be updated later
            self._pl_dataset, = ax.plot(robot_pos[0], robot_pos[1], '.', color=color)

            circle_linspace = np.linspace(0., 2*np.pi, num=360, endpoint=False)
            self.def_circle = np.transpose(np.array([np.cos(circle_linspace), np.sin(circle_linspace)]))
        
        self.h_val_toplot = np.ones(self.t_map.shape[0])
        # Grab the closest data
        is_computed = np.linalg.norm(robot_pos[:2] - self.t_map, axis=1) < sensing_rad*1.5
        map_to_plot = self.t_map[is_computed]

        """ updating the map """
        if self.N!=0: # Assign with data
            self.hpg_map,_,_,_,_=self.update_gp_computation(map_to_plot)
            self._pl_dataset.set_data(self.data_X[:,0], self.data_X[:,1])
        else: # Reset map + assing current position to plot
            self.hpg_map=np.ones(len(map_to_plot))
            self._pl_dataset.set_data([robot_pos[0]], [robot_pos[1]])    

        self.h_val_toplot[is_computed] = self.hpg_map.T[0]
        # self.hpg_map,_,_,_,_=self.update_gp_computation(self.t_map)
        

        circle_data = np.array([robot_pos[:2]]) + (self.def_circle*sensing_rad)

        if self.__prediction_plot is not None:
            self.__prediction_plot.set_array(self.h_val_toplot)
            # update circle
            self._pl_circle.set_data(circle_data[:,0], circle_data[:,1])
        else:
            self.__prediction_plot = ax.tripcolor(self.t_map[:,0],self.t_map[:,1], self.h_val_toplot,
                    vmin = -3, vmax = 3, shading='gouraud', cmap=RdGn)
            if PYSIM:
                axins1 = inset_axes(ax, width="25%", height="2%", loc='lower right')
                plt.colorbar(self.__prediction_plot, cax=axins1, orientation='horizontal', ticks=[-1,0,1])
                axins1.xaxis.set_ticks_position("top")
                # draw circle first time
                self._pl_circle, = ax.plot(circle_data[:,0], circle_data[:,1], '--', color='gray')
