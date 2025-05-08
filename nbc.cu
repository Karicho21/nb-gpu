#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

const double G = 6.674e-11;

struct simulation {
  size_t nbpart;
  
  std::vector<double> mass;

  //position
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  //velocity
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  //force
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  simulation(size_t nb)
    :nbpart(nb), mass(nb),
     x(nb), y(nb), z(nb),
     vx(nb), vy(nb), vz(nb),
     fx(nb), fy(nb), fz(nb) 
  {}
};

// CUDA kernel to reset forces
__global__ void reset_forces_kernel(double* fx, double* fy, double* fz, size_t nbpart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbpart) {
        fx[idx] = 0.0;
        fy[idx] = 0.0;
        fz[idx] = 0.0;
    }
}

// CUDA kernel to compute forces
__global__ void compute_forces_kernel(double* mass, double* x, double* y, double* z,
                                     double* fx, double* fy, double* fz,
                                     size_t nbpart, double G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    double softening = 0.1;
    double my_x = x[i];
    double my_y = y[i];
    double my_z = z[i];
    double my_fx = 0.0;
    double my_fy = 0.0;
    double my_fz = 0.0;

    for (int j = 0; j < nbpart; j++) {
        if (i == j) continue;

        double dx = x[j] - my_x;
        double dy = y[j] - my_y;
        double dz = z[j] - my_z;
        
        double dist_sq = dx*dx + dy*dy + dz*dz + softening;
        double inv_dist = rsqrt(dist_sq);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;
        
        double F = G * mass[i] * mass[j] * inv_dist3;
        
        my_fx += F * dx;
        my_fy += F * dy;
        my_fz += F * dz;
    }

    fx[i] = my_fx;
    fy[i] = my_fy;
    fz[i] = my_fz;
}

// CUDA kernel to update velocities and positions
__global__ void update_particles_kernel(double* x, double* y, double* z,
                                       double* vx, double* vy, double* vz,
                                       double* fx, double* fy, double* fz,
                                       double* mass, size_t nbpart, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbpart) {
        // Update velocities
        double inv_mass = 1.0 / mass[idx];
        vx[idx] += fx[idx] * inv_mass * dt;
        vy[idx] += fy[idx] * inv_mass * dt;
        vz[idx] += fz[idx] * inv_mass * dt;
        
        // Update positions
        x[idx] += vx[idx] * dt;
        y[idx] += vy[idx] * dt;
        z[idx] += vz[idx] * dt;
    }
}

void random_init(simulation& s) {
  std::random_device rd;  
  std::mt19937 gen(rd());
  std::uniform_real_distribution dismass(0.9, 1.);
  std::normal_distribution dispos(0., 1.);
  std::normal_distribution disvel(0., 1.);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.mass[i] = dismass(gen);

    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    s.z[i] = 0.;
    
    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
    s.vz[i] = 0.;
    s.vx[i] = s.y[i]*1.5;
    s.vy[i] = -s.x[i]*1.5;
  }

  // Normalize velocity
  double meanmass = 0;
  double meanmassvx = 0;
  double meanmassvy = 0;
  double meanmassvz = 0;
  for (size_t i = 0; i<s.nbpart; ++i) {
    meanmass += s.mass[i];
    meanmassvx += s.mass[i] * s.vx[i];
    meanmassvy += s.mass[i] * s.vy[i];
    meanmassvz += s.mass[i] * s.vz[i];
  }
  for (size_t i = 0; i<s.nbpart; ++i) {
    s.vx[i] -= meanmassvx/meanmass;
    s.vy[i] -= meanmassvy/meanmass;
    s.vz[i] -= meanmassvz/meanmass;
  }
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  // Masses in kg
  s.mass[SUN] = 1.9891e30;
  s.mass[MERCURY] = 3.285e23;
  s.mass[VENUS] = 4.867e24;
  s.mass[EARTH] = 5.972e24;
  s.mass[MARS] = 6.39e23;
  s.mass[JUPITER] = 1.898e27;
  s.mass[SATURN] = 5.683e26;
  s.mass[URANUS] = 8.681e25;
  s.mass[NEPTUNE] = 1.024e26;
  s.mass[MOON] = 7.342e22;

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496e11; // Astronomical Unit

  s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844e8};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

void dump_state(simulation& s) {
  std::cout << s.nbpart << '\t';
  for (size_t i = 0; i < s.nbpart; ++i) {
    std::cout << s.mass[i] << '\t';
    std::cout << s.x[i] << '\t' << s.y[i] << '\t' << s.z[i] << '\t';
    std::cout << s.vx[i] << '\t' << s.vy[i] << '\t' << s.vz[i] << '\t';
    std::cout << s.fx[i] << '\t' << s.fy[i] << '\t' << s.fz[i] << '\t';
  }
  std::cout << '\n';
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in(filename);
  size_t nbpart;
  in >> nbpart;
  s = simulation(nbpart);
  for (size_t i = 0; i < s.nbpart; ++i) {
    in >> s.mass[i];
    in >> s.x[i] >> s.y[i] >> s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
  if (!in.good())
    throw "Error reading file";
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
      << "usage: " << argv[0] << " <input> <dt> <nbstep> <printevery> <blocksize>" << "\n"
      << "input can be:" << "\n"
      << "a number (random initialization)" << "\n"
      << "planet (initialize with solar system)" << "\n"
      << "a filename (load from file in singleline tsv)" << "\n";
    return -1;
  }
  
  double dt = std::atof(argv[2]);
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);
  size_t blockSize = std::atol(argv[5]);
  
  simulation s(1);

  // Parse command line
  {
    size_t nbpart = std::atol(argv[1]);
    if (nbpart > 0) {
      s = simulation(nbpart);
      random_init(s);
    } else {
      std::string inputparam = argv[1];
      if (inputparam == "planet") {
        init_solar(s);
      } else {
        load_from_file(s, inputparam);
      }
    }    
  }

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();

  // Allocate device memory
  double *d_mass, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
  
  cudaMalloc(&d_mass, s.nbpart * sizeof(double));
  cudaMalloc(&d_x, s.nbpart * sizeof(double));
  cudaMalloc(&d_y, s.nbpart * sizeof(double));
  cudaMalloc(&d_z, s.nbpart * sizeof(double));
  cudaMalloc(&d_vx, s.nbpart * sizeof(double));
  cudaMalloc(&d_vy, s.nbpart * sizeof(double));
  cudaMalloc(&d_vz, s.nbpart * sizeof(double));
  cudaMalloc(&d_fx, s.nbpart * sizeof(double));
  cudaMalloc(&d_fy, s.nbpart * sizeof(double));
  cudaMalloc(&d_fz, s.nbpart * sizeof(double));

  // Copy initial data to device
  cudaMemcpy(d_mass, s.mass.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, s.x.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, s.y.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, s.z.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vx, s.vx.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy, s.vy.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz, s.vz.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fx, s.fx.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fy, s.fy.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fz, s.fz.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);

  // Calculate grid size
  size_t gridSize = (s.nbpart + blockSize - 1) / blockSize;

  for (size_t step = 0; step < nbstep; step++) {
    if (step % printevery == 0) {
      // Copy data back to host for output
      cudaMemcpy(s.fx.data(), d_fx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.fy.data(), d_fy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.fz.data(), d_fz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.x.data(), d_x, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.y.data(), d_y, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.z.data(), d_z, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.vx.data(), d_vx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.vy.data(), d_vy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.vz.data(), d_vz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      dump_state(s);
    }
  
    // Reset forces
    reset_forces_kernel<<<gridSize, blockSize>>>(d_fx, d_fy, d_fz, s.nbpart);
    
    // Compute forces
    compute_forces_kernel<<<gridSize, blockSize>>>(d_mass, d_x, d_y, d_z, 
                                                 d_fx, d_fy, d_fz, 
                                                 s.nbpart, G);
    
    // Update particles
    update_particles_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z,
                                                   d_vx, d_vy, d_vz,
                                                   d_fx, d_fy, d_fz,
                                                   d_mass, s.nbpart, dt);
    
    cudaDeviceSynchronize();
  }

  // Free device memory
  cudaFree(d_mass);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_vx);
  cudaFree(d_vy);
  cudaFree(d_vz);
  cudaFree(d_fx);
  cudaFree(d_fy);
  cudaFree(d_fz);

  // End timing and print execution time
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cerr << "Execution time: " << duration.count() << " ms\n";

  return 0;
}
