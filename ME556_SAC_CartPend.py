import numpy as np
import scipy as sci
import time
from scipy.integrate import solve_ivp
from scipy.integrate import quad, quadrature
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Angle wrapping function
def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Parameters
m_ = 1
l_ = 0.2
b_ = 0.05
g_ = 9.8
scale_factor = 1e-10
Q_ = scale_factor * np.diag([1E12, 1E1, 1E-3, 1E5])  # Weights on states
R_ = scale_factor * np.diag([1E5])  # Weights on input

def xdot(t, x, u):
    theta = x[0]
    theta_dot = x[1]
    cart_pos = x[2]
    cart_vel = x[3]
    
    theta_ddot = g_/l_ * np.sin(theta) + u * np.cos(theta) - b_/(m_*l_**2)*theta_dot
    x_ddot = u
    
    return np.array([theta_dot, theta_ddot, cart_vel, x_ddot])
def h(x):
    H = np.array([0,-np.cos(x[0])/l_,0,1])
    return H.T

stage_cost = []
stage_cost_all = []
def stage(x, u):
    for i in range(N):
        stage_cost = (x[:,i].T @ Q_ @ x[:,i] + u * R_ * u)/2
        stage_cost_all.append(stage_cost)
    return np.squeeze(stage_cost_all, axis=2)


def J(x, u, t1, t2):
    # Define a callable for the integrand
    def integrand(t):
        # Interpolate x and u based on t if needed (e.g., linear interpolation)
        index = round((t - t1) / (t2 - t1) * (x.shape[1] - 1))  # Map t to index
        x_t = x[:, index]  # Get x at time t
        u_t = u  # Assuming constant control input, or modify if u varies with time
        return (x_t.T @ Q_ @ x_t + u_t.T @ R_ @ u_t) / 2

    # Perform integration
    fullcost, err = quad(integrand, t1, t2, limit = 10000)
    return fullcost

def rhodot(t, rho, xall, u, tnow):
    x = xall[:,round((t-tnow)/dt)-1]
    Dxf1 = np.array([0, 1, 0, 0])
    Dxf2 = np.array([g_/l_*np.cos(x[0]) + u*np.sin(x[0])/l_, b_/(m_*l_**2), 0, 0])
    Dxf3 = np.array([0, 0, 0, 1])
    Dxf4 = np.array([0, 0, 0, 0])
    Dxf = np.vstack([Dxf1, Dxf2, Dxf3, Dxf4])
    rhodot = -2 * x.T @ Q_ - Dxf.T @ rho
    return rhodot

# SAC initializations
delJmin = 0.  # Initial Cost
tcurr = 0.  # Current time
dt = 1/1000  # Time step
tcalc = 1.  # Control duration
T =  1 # Prediction horizon 
ts = dt  # Sampling time
u1 = np.array([0.0])  # Initial input
gamma = -15  # Negative value from [-15, -1]
alpha_d = -1  # Very negative value
beta = 0.55
w = 0.5  # Scale factor [0,1]
kmax = 5  # Maximum backtracking iterations
xinit = np.array([np.pi, 0, 0, 0]) # Starting pendulum at bottom position
N = round(T/dt)
xlist = []
ulist = []

while tcurr < np.inf:
    #Stopping criteria to allow for animation to run
    if len(xlist) >= 5000:
        break
    [t0, tf] = [tcurr, tcurr + T]
    t_forw = np.linspace(t0,tf,N)
    t_back = np.linspace(tf,t0,N)

    xdot_sol = solve_ivp(xdot, [t0, tf], xinit, args = (u1[0],), t_eval=t_forw)  # Simulating x for current horizon
    xi = xdot_sol.y  # Nx4x1
    for i in range(N):
         xi[:,i][0] = wrap(xi[:,i][0])

    rhodot_sol = solve_ivp(rhodot, [tf, t0], [0,0,0,0], args = (xi, u1[0], tcurr), t_eval=t_back)
    rhoback = [np.reshape(rhodot_sol.y[:,i],(4,1)) for i in range(N)] # Reshape rho for later matrix algebra
    rhoi = rhoback[::-1] # Reverse rho since the integration was backwards
    J1init = J(xi, u1, t0, tf)  # Computing initial cost, not using a terminal cost
    alpha_d = gamma * J1init

    # Initializing lists
    f1_list = []
    hx_list = []
    biglam_list = []
    u2star_list = []
    # Calculating nominal dynamics and u2star
    for i in range(N):
        xi_i = xi[:,i]  # shape (4,1)
        # Extract the i-th 4x1 vector from rhoi
        rhoi_i = rhoi[i]  # shape (4,1)
        # Compute f1 = xdot(0, xi_i, u1)
        f1 = xdot(0, xi_i, u1[0])  # Nominal dynamics, needed for dJdlam
        hx = h(xi_i)
        # Compute biglam = f1.T @ rhoi_i @ rhoi_i.T @ f1 (will be 1x1 array)
        biglam = hx.T @ rhoi_i @ rhoi_i.T @ hx
        u2star = (biglam * u1 + hx.T @ rhoi_i * alpha_d)*np.linalg.inv(biglam + R_)
        # Append results for this iteration
        f1_list.append(f1)          # Each f1 is a (4,1) array
        hx_list.append(hx)
        biglam_list.append(biglam)  # Scalar
        u2star_list.append(u2star)  # Scalar

    # Converting lists into arrays
    f1_all = np.array(f1_list)      # shape (4, N)
    hx_all = np.array(hx_list)
    biglam_all = np.array(biglam_list)   # shape (N,)
    u2star_all = np.array(u2star_list)   # shape (N,)

    # Calculating dynamics for all possible application times of u2star
    f2_list = []
    for j in range(N):
        xi_j = xi[:,j]
        u2star_j = u2star_all[j]
        f2 = xdot(0, xi_j, u2star_j[0][0])
        f2_list.append(f2)
    # Converting list into array
    f2_all = np.array(f2_list)

    # Calculating dJdlam, used in part of calculations to decide when to act
    dJdlam_list = []
    for i in range(N):
        dJdlam = rhoi[i].T @ (f2_all[i] - f1_all[i])
        dJdlam_list.append(dJdlam)
    dJdlam_all = np.array(dJdlam_list)

    # Initializing Jtau
    Jtau_list = []
    # Finding when to act by analyzing when Jtau is the lowest
    for i in range(N):
        Jtau = np.linalg.norm(u2star_all[i])+dJdlam_all[i] + dt**beta
        Jtau_list.append(Jtau)
    if xinit[0] != 0:
        Jtau_list = Jtau_list[:-1]
    else:
        Jtau_list = Jtau_list
    Jtau_all = np.array(Jtau_list) # Nx1
    tau_ind = Jtau_list.index(min(Jtau_list))
    tau = tcurr + tau_ind*dt
    #Using tau index to find corresponding single optimal input and then saturating it based on constraints
    u2star_clip = np.clip(u2star_all[tau_ind], -100, 100)
    print('Tau index is:')
    print(tau_ind)
    print('U2star clip is:')
    print(u2star_clip)

    k = 0 # Initializing backtracking iterations
    J1new = np.inf # Intializing J1new
    delJmin = -0.01*J1init
    while J1new-J1init > delJmin and k <= kmax:
        lam = w * dt
        [tau0, tauf] = [tau - lam / 2, tau + lam / 2]
        xdot_temp = solve_ivp(xdot, [t0, tf], xinit, args = (u2star_clip[0][0],), t_eval=t_forw)
        xinit_temp = xdot_temp.y
        for i in range(N):
            xinit_temp[:,i][0] = wrap(xi[:,i][0])
        J1new = J(xinit_temp, u2star_clip[0], t0, tf)
        k += 1
    u1 = u2star_clip[0]
    for i in range(N):
        xlist.append(xinit_temp[:,i])
        ulist.append(u1)
    tcurr = tcurr + ts
    xinit = xlist[-1]


# After your simulation loop ends or after you've gathered enough data:
xarray = np.array(xlist)  # shape (num_steps, 4)
# xarray[:,0] = rod angle (theta)
# xarray[:,1] = rod angular velocity (theta_dot)
# xarray[:,2] = cart position (x)
# xarray[:,3] = cart velocity (x_dot)

ulist = np.array(ulist)  # shape (num_steps, )

# Ensure that xarray and ulist have the same number of time steps
assert xarray.shape[0] == ulist.shape[0], "Mismatch in number of time steps between states and control inputs."

# Create a time array based on the number of steps and the time step (dt)
num_steps = xarray.shape[0]
time = np.arange(num_steps) * dt  # time in seconds

# Define state labels for clarity in plots
state_labels = ['Theta (rad)', 'Theta_dot (rad/s)', 'Cart Position (m)', 'Cart Velocity (m/s)']

# Define colors for each state for consistency across subplots
state_colors = ['blue', 'orange', 'green', 'red']

# Create a figure and a set of subplots (4 rows, 1 column)
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Adjust the layout to prevent overlap
fig.subplots_adjust(hspace=0.4)

# Plot each state in its respective subplot
for i in range(4):
    axs[i].plot(time, xarray[:, i], color=state_colors[i], label=state_labels[i])
    axs[i].set_ylabel(state_labels[i], fontsize=12)
    axs[i].legend(loc='upper right', fontsize=10)
    axs[i].grid(True)
    axs[i].tick_params(axis='y', labelsize=10)

# Set the x-axis label on the last subplot
axs[-1].set_xlabel('Time (s)', fontsize=12)
axs[-1].tick_params(axis='x', labelsize=10)

# Figure Title
fig.suptitle('System States Over Time', fontsize=16)

# Display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""
!!!ANIMATION!!!

# Plotting parameters
cart_width = 0.4
cart_height = 0.2
length = l_
gravity = g_

fig, ax = plt.subplots()

# Set plot limits
x_min = np.min(xarray[:,2]) - 1.0
x_max = np.max(xarray[:,2]) + 1.0
y_min = -length - 0.5
y_max = cart_height + length + 0.5

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])

if x_min <= -5 or x_max >= 5:
    ax.set_aspect(3)
else:
    ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('Horizontal Position')
ax.set_ylabel('Vertical Position')
ax.set_title('Inverted Pendulum on a Cart')

# Elements to draw
cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
ax.add_patch(cart_patch)

rod_line, = ax.plot([], [], lw=2, color='red')
bob_patch = plt.Circle((0, 0), 0.05, fc='red', ec='black')
ax.add_patch(bob_patch)

def init():
    cart_patch.set_xy((xarray[0,2] - cart_width/2, 0))
    rod_line.set_data([], [])
    bob_patch.center = (xarray[0,2] + length*np.sin(xarray[0,0]), length*np.cos(xarray[0,0]) + cart_height)
    return cart_patch, rod_line, bob_patch

def animate(i):
    # Extract states at step i
    theta = xarray[i,0]
    x_cart = xarray[i,2]
    
    # Update cart position
    cart_patch.set_xy((x_cart - cart_width/2, 0))
    
    # Pendulum coordinates
    pivot_x = x_cart
    pivot_y = cart_height
    
    bob_x = pivot_x + length * np.sin(theta)
    bob_y = pivot_y + length * np.cos(theta)  # Change from -cos(theta) to +cos(theta)
    
    rod_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
    bob_patch.center = (bob_x, bob_y)
    
    return cart_patch, rod_line, bob_patch

ani = animation.FuncAnimation(fig, animate, frames=len(xarray), 
                              interval=1, blit=True, init_func=init, repeat=False)

plt.show()
"""
