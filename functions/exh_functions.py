import sys
import numpy as np
np.set_printoptions(threshold=np.inf)

#Determine the path of the repository to set paths correctly below.
repo_path = r'C:\Users\Sofia\Documents\Sofia\Noddy'
sys.path.insert(0, repo_path)

import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.experiment
importlib.reload(pynoddy.experiment)
import copy
import pandas as pd
import os

#function for getting noddy model coordinates by Kaifeng Gao
def ExtractCoords(hist_moment, lith,res):
    
    # Compute noddy model for history file
    temp_hist = 'synthetic_outputs/temp_hist.his'
    temp_out = 'synthetic_outputs/temp_out'
    hist_moment.write_history(temp_hist)
    pynoddy.compute_model(temp_hist, temp_out, 
                          noddy_path = r'C:\Users\Sofia\pynoddy\noddyapp\noddy_win64.exe')
    N1 = pynoddy.output.NoddyOutput(temp_out)
    #N1.plot_section('y', litho_filter = lith)
    
    #num_rock = N1.n_rocktypes   #number of rock types
    sum_node = []   #stack the nodes in each interface
    num_list = []   #output the node number in each interface
    
    if isinstance(lith, list):
        #litho_list = list(range(1, num_rock+1))  ##the lithoID in noddy begin at 2, lithoID could be a lis
        for i in lith:
            rockID = i*2  #because points = np.absolute(points/2), multiply 2 in advance
            out = N1.get_surface_grid(lithoID=[i],res=res)  #here can change the sampling number by changing res
            listx = out[0][0][i]   #out[0]：get_surface_grid, only choose first direction；out[0][0]: x values in x-grid
            listy = out[0][1][i]   #y values
            listz = out[0][2][i]
            xx = sum(listx, [])  #multi-rows to one row
            yy = sum(listy, [])
            zz = sum(listz, [])
            num_node = 0
            for j in range(len(xx)):
                #if xx[j] <-1 and yy[j]<-1 and zz[j]<-1:
                sum_node.append([xx[j],yy[j],zz[j],rockID])
                #num_node +=1
            #num_list.append(num_node)

        
    #get_surface_grid function will get negative value, and twice the value, don't know the reason
    points = np.array(sum_node)
    points = np.absolute(points/2)
    np.set_printoptions(precision=2) #two decimal places
    np.set_printoptions(suppress=True) #don't use scientific notation

    return points, num_list, N1, temp_hist  

def get_indicator_grid(self, lithoID,indicator, **kwds):
    
    import numpy.ma as ma
    cube_size = self.xmax / self.nx
    res = kwds.get('res', 2)
    if not type(lithoID) is list:
        lithoID = [lithoID]
    sx = {}
    sy = {}
    sz = {}
    sc = {}
    
    if indicator == 'X':
        loop1_range, loop2_range, loop3_range = self.ny, self.nz, self.nx
    elif indicator == 'Y':
        loop1_range, loop2_range, loop3_range = self.nx, self.nz, self.ny
    elif indicator == 'Z':
        loop1_range, loop2_range, loop3_range = self.nx, self.ny, self.nz
    
    # get surface locations in x direction
    for loop1 in range(0, loop1_range, res):
        # start new line
        for i in lithoID:
            if i not in sx:  # create list
                sx[i] = []
                sy[i] = []
                sz[i] = []
                if (hasattr(self, 'rock_colors')):
                    sc[i] = self.rock_colors[i]
                else:
                    sc[i] = i
            sx[i].append([])
            sy[i].append([])
            sz[i].append([])
        # fill in line
        for loop2 in range(0, loop2_range, res):
            # drill down filling surface info
            found = []
            for loop3 in range(0, loop3_range - 1):
                if indicator == 'X':
                    if (self.block[loop3][loop1][loop2] != self.block[loop3+1][loop1][loop2]) and self.block[loop3][loop1][loop2] in lithoID:
                        key = self.block[loop3][loop1][loop2]
                    # add point
                        sx[key][-1].append(loop3 * cube_size)
                        sy[key][-1].append(loop1 * cube_size)
                        sz[key][-1].append(loop2 * cube_size)
                    # remember that we've found this
                        found.append(key)
                elif indicator == 'Y':
                    if (self.block[loop1][loop3][loop2] != self.block[loop1][loop3+1][loop2]) and self.block[loop1][loop3][loop2] in lithoID:
                        key = self.block[loop1][loop3][loop2]
                    # add point
                        sx[key][-1].append(loop1 * cube_size)
                        sy[key][-1].append(loop3 * cube_size)
                        sz[key][-1].append(loop2 * cube_size)
                elif indicator == 'Z':
                    if (self.block[loop1][loop2][loop3] != self.block[loop1][loop2][loop3+1]) and self.block[loop1][loop2][loop3] in lithoID:
                        key = self.block[loop1][loop2][loop3]
                    # add point
                        sx[key][-1].append(loop1 * cube_size)
                        sy[key][-1].append(loop2 * cube_size)
                        sz[key][-1].append(loop3 * cube_size)    
            # check to see if anything has been missed(and hence we should start a new line segment)
            for i in lithoID:
                if not i in found:
                    sx[i].append([])  # new list
                    sy[i].append([])
                    sz[i].append([])

    xlines = (sx, sy, sz, sc)
    sx = {}
    sy = {}
    sz = {}
    sc = {}
    # get surface locations in y direction
    for loop2 in range(0, loop2_range, res):
        # start new line
        for i in lithoID:
            if i not in sx:  # create list
                sx[i] = []
                sy[i] = []
                sz[i] = []
                if (hasattr(self, 'rock_colors')):
                    sc[i] = self.rock_colors[i]
                else:
                    sc[i] = i
            sx[i].append([])
            sy[i].append([])
            sz[i].append([])
        # fill in line
        for loop1 in range(0, loop1_range, res):
            # drill down filling surface info
            found = []
            for loop3 in range(0, loop3_range - 1):
                if indicator == 'X':
                    if (self.block[loop3][loop1][loop2] != self.block[loop3+1][loop1][loop2]) and self.block[loop3][loop1][loop2] in lithoID:
                        key = self.block[loop3][loop1][loop2]
                    # add point
                        sx[key][-1].append(loop3 * cube_size)
                        sy[key][-1].append(loop1 * cube_size)
                        sz[key][-1].append(loop2 * cube_size)
                    # remember that we've found this
                        found.append(key)
                elif indicator == 'Y':
                    if (self.block[loop1][loop3][loop2] != self.block[loop1][loop3+1][loop2]) and self.block[loop1][loop3][loop2] in lithoID:
                        key = self.block[loop1][loop3][loop2]
                    # add point
                        sx[key][-1].append(loop1 * cube_size)
                        sy[key][-1].append(loop3 * cube_size)
                        sz[key][-1].append(loop2 * cube_size)
                elif indicator == 'Z':
                    if (self.block[loop1][loop2][loop3] != self.block[loop1][loop2][loop3+1]) and self.block[loop1][loop2][loop3] in lithoID:
                        key = self.block[loop1][loop2][loop3]
                    # add point
                        sx[key][-1].append(loop1 * cube_size)
                        sy[key][-1].append(loop2 * cube_size)
                        sz[key][-1].append(loop3 * cube_size)    
            for i in lithoID:
                if not i in found:  # line should end
                    sx[i].append([])  # add line end
                    sy[i].append([])
                    sz[i].append([])
    ylines = (sx, sy, sz, sc)
    return (xlines, ylines)

def ExtractCoordsSimple(output, lith, res):
    
    # Compute noddy model for history file
    sum_node = []
    if isinstance(lith, list):
        #litho_list = list(range(1, num_rock+1))  ##the lithoID in noddy begin at 2, lithoID could be a lis
        for i in lith:
            rockID = i*2  #because points = np.absolute(points/2), multiply 2 in advance
            out = output.get_surface_grid(lithoID=[i], res=res)  #here can change the sampling number by changing res
            listx = out[0][0][i]   #out[0]：get_surface_grid, only choose first direction；out[0][0]: x values in x-grid
            listy = out[0][1][i]   #y values
            listz = out[0][2][i]
            xx = sum(listx, [])  #multi-rows to one row
            yy = sum(listy, [])
            zz = sum(listz, [])
            for j in range(len(xx)):
                sum_node.append([xx[j],yy[j],zz[j],rockID])
        
    #get_surface_grid function will get negative value, and twice the value, don't know the reason
    points = np.array(sum_node)
    points = np.absolute(points/2)
    np.set_printoptions(precision=2) #two decimal places
    np.set_printoptions(suppress=True) #don't use scientific notation

    return points

def disturb_percent(event, prop, percent = 30):
    """Disturb the property of an event by a given percentage, assuming a normal distribution"""
    ori_val = event.properties[prop]
    new_val = np.random.randn() * percent/100. * ori_val + ori_val
    event.properties[prop] = new_val
    
    return new_val
    
def disturb_value(event, prop, stdev):
    """Disturb the property of an event by a given stdev, assuming a normal distribution"""
    ori_val = event.properties[prop]
    new_val = np.random.normal(ori_val, stdev)
    event.properties[prop] = new_val
    
    return new_val

def disturb_value_rounded(event, prop, stdev):
    """Disturb the property of an event by a given stdev, assuming a normal distribution"""
    ori_val = event.properties[prop]
    new_val = round(np.random.normal(ori_val, stdev),-2)
    event.properties[prop] = new_val
    
    return new_val

# function for disturbing the model once
def disturb(PH_local, std_list, ndraw):
    data = []
    for event_name, event in PH_local.events.items():
        if isinstance(event, pynoddy.events.Fault):
            new_slip = disturb_value(event, 'Slip', std_list[0])
            new_amp = disturb_value(event, 'Amplitude', std_list[1])
            #new_x = disturb_value(event, 'X', std_list[2])
            #new_dip = disturb_percent(event, 'Dip', percent=5)
            #new_dipdir = disturb_percent(event, 'Dip Direction', std_list[3])
            #new_pitch = disturb_percent(event, 'Pitch', percent=5)
            #new_z = disturb_value(event, 'Z', 75)
            data.append([event_name, new_slip, new_amp, new_x, new_dipdir, ndraw])
    
    columns = ['Event', 'New Slip', 'New Amplitude', 'New X', 'New Dip Direction','nDraw']
    df = pd.DataFrame(data, columns=columns)
    return data, df

def exhumationComplex(history, lith, res = 8, interval = 50, upperlim = 0):
    
    """function for estimating the exhumation (vertical movement) from a noddy history. Arguments:
            lith: lith id of the dyke or item used to track the movement
            res: 
            interval: sampling interval in the z direction 
            upperlim: limit up to which sampling is performed.
            """
    
    new_z = upperlim - 1
    n_sample = 0
    
    coords = []
    hist_copy = copy.deepcopy(history)
    
    while new_z < upperlim:
        n_sample += 1
        
        for event in hist_copy.events.values():
            if isinstance(event, pynoddy.events.Dyke):
                old_z = event.properties['Z']
                new_z = old_z + interval
                event.properties['Z'] = new_z
                print(new_z)
    
                points,_,N1,_ = ExtractCoords(hist_copy, lith = lith, res = res) #make sure that the history is at least cube size 100.
                
                try:
                    x = points[...,0]
                    y = points[...,1]
                    z = points[...,2]
                except IndexError:
                    continue
                
                #correct for weird noddy coordinates
                #real_z = points[0][2] #select the Z value of the first row.
                real_z = points[...,2].min() #select the minimum z value - that's the original depth
                exhumation =  z - real_z
        
        for j in range(len(x)):    
            coords.append([n_sample,x[j],y[j],z[j],exhumation[j]])
        
        #coords = np.array(coords)
            
    return coords, N1, hist_copy

def calc_new_position(hist, diff, og_depths, lith_list,samples):
    samples_noddy_pos = []
    for i in lith_list:
        p,_,out,new_hist = ExtractCoords(hist, lith = [i], res = 1)
        t = p[...,2].min()
        z = (t*1000) / 3681.39
        samples_noddy_pos.append(z)
    
    if len(lith_list) > 1:
        proposed_exhumation = [x - y - z for x,y,z in zip(samples_noddy_pos, diff, og_depths)]
    else:
        proposed_exhumation = samples_noddy_pos - diff - og_depths
    samples['exhumation'] = proposed_exhumation
    return samples, samples_noddy_pos, new_hist 

def disturb_property(PH_local, event_list, prop_list, std_list):
    data = []
    for i in event_list:
        event_data = [i]
        for j, prop in enumerate(prop_list):
            new_param = disturb_value(PH_local.events[i], prop_list[j], std_list[j])
            rounded_param = round(new_param, -2)
            event_data.append(rounded_param)
            
        data.append(event_data)
    col = ['event_name'] + prop_list
    df = pd.DataFrame(data, columns = col)
    
    return data, df

def create_pdf(mean, std_dev):
    def pdf(x):
        coeff = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = - ((x - mean) ** 2) / (2 * std_dev ** 2)
        return coeff * np.exp(exponent)
    return pdf

def prior_dist(og_params,proposed_params,std_list):
    log_prior_prob = 1.0
    for i in range(len(og_params)):
        for j in range(len(std_list)):
            #print(og_params[i][j+1], std_list[j], proposed_params[i][j+1])
            pdf = create_pdf(og_params[i][j+1], std_list[j])
            log_prior_prob *= pdf(proposed_params[i][j+1])
    return log_prior_prob

def cont_likelihood(mu, sigma, x):
    #mu is the model prediciton, sigma is the errorof the data and x is the observed data
    # Calculate the likelihood using the Gaussian PDF
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pdf

def synthetic_likelihood(exhumation_df, synthetic_data, sigma):
    #The bigger the sigma, the bigger the likelihood
    likelihood = 1.0
    
    for i in range(len(synthetic_data)):
        modeled_value = exhumation_df.iloc[i]['exhumation']
        data = synthetic_data[i][1]
        
        like_value = cont_likelihood(modeled_value, sigma, data)
        print(like_value)
        likelihood *= like_value
    
    return likelihood