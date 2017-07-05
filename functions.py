def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Function source:
    https://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker
    
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection

#===================================================================

def deposit_particles(theta,N,L,r):
    """
    Creating the ballistic deposition simulation.
    Inputs are
    - theta: inclination angle
    - N: total number of deposited particles
    - L: box width
    - r: particle size
    
    Outputs particle center coordinates.
    
    """
    
    from numpy import sin,cos,tan,array,pi,sqrt,fmod,argsort
    from numpy.random import rand
    
    # envelope calculation
    from scipy.stats import binned_statistic
    
    # binary index search in ordered lists
    # https://docs.python.org/2/library/bisect.html
    from bisect import bisect_left,bisect_right
    
    # for displaying progress bar
    from tqdm import trange,tqdm_notebook


    h=50*N/L # milyen magasrol esnek a reszecskek - vegtelenre allitom
    
    # elore kiszamolt szogfuggvenyek
    t=tan(float(theta)/180*pi)
    s=sin(float(theta)/180*pi)
    c=cos(float(theta)/180*pi)

    # tarolas
    xs=list() # xs koordinatak (ferde koordinatarendszer)
    ys=list() # ys koordinatak (ferde koordinatarendszer), rendezetten tartjuk
    txt=list()

    # ezt at kellene alakitani! nem kozben szamoltatni, hanem ha elore legeneraltam egy csomo reszecsket
    # akkor utana adott n-ig lehessen visszakerni az envelope-ot
    # sokkal gyorsabb lesz

    for particle in trange(N):
        # sorsolni egy kezdoppoziciot
        p=rand()*L

        # megkeresni azokat, akik az egyenes trajektoriatol +/- 2r tavolsagban vannak
        ll=p-2*float(r)/s
        rr=p+2*float(r)/s
        indlist=[]

        if (ll>0) and (rr<L): # ha nem erem el a hatarokat
            typ=0
            left=bisect_left(ys,ll)
            right=bisect_right(ys,rr)
            indlist+=range(left,right)
            bound=0
        elif (ll<0) and (rr<L): # ha elerem a bal hatarfeluletet, periodikus hatarfeltetel
            typ=-1
            left=bisect_right(ys,ll+L)
            indlist+=range(left,len(ys))
            bound=len(indlist)
            right=bisect_right(ys,rr)
            indlist+=range(0,right)
        elif (rr>L) and (ll>0): # ha elerem a jobb hatarfeluletet, periodikus hatarfeltetel
            typ=+1
            right=bisect_left(ys,rr-L)
            indlist+=range(0,right)
            bound=len(indlist)
            left=bisect_left(ys,ll)
            indlist+=range(left,len(ys))

        if len(indlist)>0:
            # korrigalom a szamolas erejeig a periodikus hataron levo koordinatakat
            ys_corrected=[]
            for k in range(len(indlist)):
                if (k<bound):
                    ys_corrected.append(array(ys)[indlist][k]+typ*L)
                else:
                    ys_corrected.append(array(ys)[indlist][k])
            ys_corrected=array(ys_corrected)
            xs_corrected=array(xs)[indlist]

            # first possible elements
            m=min(xs_corrected)
            mask=(xs_corrected-m)<=abs(2*r*(1/t+1/c))
            ys_corrected=ys_corrected[mask]
            xs_corrected=xs_corrected[mask]

            dy=array(ys_corrected)-p # a beeses egyenesetol vett elteres a ferdeszogu rendszerben
            # a savba eso reszecskek uj koordinatai a ferdeszogu rendszerben
            b=-(2*dy*c+2*xs_corrected)
            const=xs_corrected**2+dy**2-4*r**2+2*dy*xs_corrected*c
            d=sqrt(b**2-4*const)
            q=(-b-d)/2
            # ezek kozul a minimalisnal all meg eloszor a reszecske
            qq=min(q)

            # uj koordinatak
            xn=qq
            yn=p

            # insert new coordinates
            ind=bisect_left(ys,yn)
            xs.insert(ind,xn)
            ys.insert(ind,yn)
            txt.insert(ind,particle)
        else:
            # ha a szubsztratba utkoznek bele a reszecskek eloszor
            # new coordinates
            xn=float((h-r))/s
            yn=p

            # insert new coordinates
            ind=bisect_left(ys,yn)
            xs.insert(ind,xn)
            ys.insert(ind,yn)
            txt.insert(ind,particle)
            
    # transzformacio Descartes-koordinatakba
    x=fmod(array(xs)*c+array(ys),L) # amik kilognak, betesszuk (0,L) koze
    y=h-array(xs)*s
    txt=array(txt) # reszecskek leerkezesi sorszama
    
    li=argsort(txt)
    x=fmod(x[li]+L,L)
    y=y[li]
    
    return x,y

def create_envelope(x,y,n,r=1,L=100,sampling_factor=100):
    """
    Function for calculating the surface of the deposit.
    Deposit coordinates are given by x and y.
    It is possible to only use the first n fallen particles.
    
    The function samples points from the boundaries of the particles,
    then sampled points are binned according to x, and the maximum y
    is taken in each bin.
    
    Inputs:
        - x and y coordinates of deposited particles
        - n number of particles to use from x and y
        - r radius of deposited particle
        - L length of substrate
        - sampling factor: how many points to take from the particle boundaries
    """
    
    from scipy.stats import binned_statistic
    from numpy import linspace,pi,sin,cos,floor,isnan
    
    sample_x=[]
    sample_y=[]
    
    # one small sampled circle
    t=linspace(0,2*pi,sampling_factor)
    tx=r*sin(t)
    ty=r*cos(t)
    
    # adding sampled circle to all particle coordinates
    for i in range(n):
        sample_x+=list(x[i]+tx)
        sample_y+=list(y[i]+ty)
        
    # binning by x, taking maximum by y
    ymax,be,bn=binned_statistic(sample_x,sample_y,statistic=max,bins=floor(5*float(L)/r),range=(0,L))
    
    # return envelope x,y, all sampled x,y - that was for debugging purposes
    ymax[isnan(ymax)]=0
    return be[1:],ymax

def create_envelope_selected(x,y,nlist,r=1,L=100,sampling_factor=100):
    """
    Function for calculating the surface of the deposit.
    Deposit coordinates are given by x and y.
    This function only uses the particles given in nlist.
    
    The function samples points from the boundaries of the given particles,
    then sampled points are binned according to x, and the maximum y
    is taken in each bin.
    
    Inputs:
        - x and y coordinates of deposited particles
        - nlist indices of particles to use from x and y
        - r radius of deposited particle
        - L length of substrate
        - sampling factor: how many points to take from the particle boundaries
    """
    
    from scipy.stats import binned_statistic
    from numpy import linspace,pi,sin,cos,floor,isnan
    
    sample_x=[]
    sample_y=[]
    
    # one small sampled circle
    t=linspace(0,2*pi,sampling_factor)
    tx=r*sin(t)
    ty=r*cos(t)
    
    # adding sampled circle to all particle coordinates
    for i in nlist:
        sample_x+=list(x[i]+tx)
        sample_y+=list(y[i]+ty)
        
    # binning by x, taking maximum by y
    ymax,be,bn=binned_statistic(array(sample_x),array(sample_y),statistic=max,bins=floor(5*float(L)/r),range=(0,L))
    
    # return envelope x,y, all sampled x,y - that was for debugging purposes
    ymax[isnan(ymax)]=0
    return be[1:],ymax