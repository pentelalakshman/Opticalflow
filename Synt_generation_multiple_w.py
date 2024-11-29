# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:10:39 2024

@author: laksh
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Parameters
frame_size = (512, 512)  # Frame size (width, height)
num_frames = 150  # Number of frames to generate
num_particles = 2  # Number of particles
initial_velocity = 4  # Maximum initial vertical velocity (pixels per frame)
particle_radius = 10  # Radius of particles
collision_distance = 2 * particle_radius  # Distance for collision handling
gravity = 0.2  # Gravitational acceleration (pixels/frame^2)
restitution = 0.8  # Coefficient of restitution (bounciness factor)
damping = 0.99  # Damping factor to simulate air resistance

np.random.seed(42)  # For reproducibility

# Initial positions (x, y) and velocities (vx, vy)
x_positions = np.linspace(particle_radius, frame_size[0] - particle_radius, num_particles)
np.random.shuffle(x_positions)
positions = np.column_stack((x_positions, np.full(num_particles, particle_radius))).astype(float)

velocities = np.column_stack((
    np.random.uniform(-2, 2, num_particles),  # Random horizontal velocity
    -np.random.uniform(0.5, initial_velocity, num_particles)  # Random downward velocity
))

# Output paths
output_video_path = "particles_with_physics_2p.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, frame_size)  # 20 FPS

card_file_path = "particles_with_physics_2p.txt"
with open(card_file_path, 'w') as card:
    card.write('Frame No.\t' + '\t'.join([f"Particle_{i+1} (x, y, vx, vy)" for i in range(num_particles)]) + '\n')

# Visualization parameters
cmap = plt.get_cmap('jet')
norm = mcolors.Normalize(vmin=-initial_velocity, vmax=initial_velocity)

# Simulation loop
for frame_num in range(num_frames):
    # Apply gravity and update positions
    velocities[:, 1] += gravity  # Add gravitational acceleration
    positions += velocities

    # Handle boundary collisions
    for i in range(num_particles):
        # Bottom boundary
        if positions[i, 1] >= frame_size[1] - particle_radius:
            velocities[i, 1] = -velocities[i, 1] * restitution  # Reverse and reduce velocity
            positions[i, 1] = frame_size[1] - particle_radius - 1  # Clamp position

        # Top boundary
        elif positions[i, 1] <= particle_radius:
            velocities[i, 1] = -velocities[i, 1] * restitution  # Reverse velocity
            positions[i, 1] = particle_radius

        # Horizontal boundaries (left/right)
        if positions[i, 0] <= particle_radius:
            velocities[i, 0] = -velocities[i, 0] * restitution  # Reverse horizontal velocity
            positions[i, 0] = particle_radius
        elif positions[i, 0] >= frame_size[0] - particle_radius:
            velocities[i, 0] = -velocities[i, 0] * restitution  # Reverse horizontal velocity
            positions[i, 0] = frame_size[0] - particle_radius - 1  # Clamp position

    # Apply damping
    velocities *= damping

    # Handle particle-particle collisions
# Particle-particle collision handling with momentum exchange and random perturbation
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= collision_distance:
                # Compute the collision normal and relative velocity
                collision_normal = (positions[i] - positions[j]) / distance
                relative_velocity = velocities[i] - velocities[j]
                normal_velocity = np.dot(relative_velocity, collision_normal)
    
                if normal_velocity < 0:  # Only handle if particles are moving toward each other
                    # Momentum exchange using 1D elastic collision in the normal direction
                    impulse = -2 * normal_velocity / (1 / particle_radius + 1 / particle_radius)
                    velocities[i] += impulse * collision_normal / particle_radius
                    velocities[j] -= impulse * collision_normal / particle_radius
    
                    # Add small random perturbations to prevent clustering
                    perturbation = np.random.uniform(-0.2, 0.2, size=2)
                    velocities[i] += perturbation
                    velocities[j] -= perturbation


    # Visualization: Contour map and particles
    velocity_map = np.zeros(frame_size)
    for i in range(num_particles):
        pos = np.round(positions[i]).astype(int)
        velocity_map[pos[1], pos[0]] = velocities[i, 1]  # Use vertical velocity for the contour

    plt.figure(figsize=(5.12, 5.12), dpi=150)
    contour = plt.contourf(velocity_map, cmap='jet', levels=np.linspace(np.min(velocity_map), np.max(velocity_map), 100))
    plt.colorbar(contour, label='Vertical Velocity (pixels/frame)')
    plt.title(f'Frame {frame_num + 1} - Vertical Velocity Contour')
    plt.gca().invert_yaxis()
    plt.axis('off')

    for i in range(num_particles):
        pos = np.round(positions[i]).astype(int)
        color = cmap(norm(velocities[i, 1]))
        plt.scatter(pos[0], pos[1], color=color)

    plt.show()

    # Write video frame
    frame = np.zeros(frame_size)
    for idx, pos in enumerate(np.round(positions).astype(int)):
        cv2.circle(frame, (pos[0], pos[1]), particle_radius, (255, 255, 255), -1)

    video_writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    # Write data to file
    with open(card_file_path, 'a') as card:
        row = f"Frame {frame_num + 1}\t"
        row += '\t'.join([f"{pos[0]:.1f}, {pos[1]:.1f}, {vel[0]:.2f}, {vel[1]:.2f}"
                          for pos, vel in zip(positions, velocities)])
        card.write(row + '\n')

video_writer.release()
print(f"Video saved to {output_video_path}")
print(f"Particle data saved to {card_file_path}")
