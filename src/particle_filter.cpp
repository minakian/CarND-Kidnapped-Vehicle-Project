/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 100;

  particles.resize(num_particles);
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0,1);

  for(int i=0; i<num_particles; i++){

    particles[i].x = x + std[0] * distribution(generator);
    particles[i].y = y + std[1] * distribution(generator);
    particles[i].theta = theta + std[2] * distribution(generator);
//    particles[i].weight = 1./num_particles;
    particles[i].weight = 1.;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0,1);
  if(yaw_rate == 0){
    yaw_rate = 0.0000001;
  }
  for(int i = 0; i < num_particles; i++){
    double x_f;
    double y_f;
    double theta_f;
    double temp;

    Particle &particle = particles[i];

    theta_f = particle.theta + yaw_rate * delta_t;
    x_f = particle.x + (velocity/yaw_rate)*(sin(theta_f) - sin(particle.theta));
    y_f = particle.y + (velocity/yaw_rate)*(cos(particle.theta) - cos(theta_f));

//    std::normal_distribution<double> distribution_x(x_f,std_pos[0]);
//    std::normal_distribution<double> distribution_y(y_f,std_pos[1]);
//    std::normal_distribution<double> distribution_theta(theta_f,std_pos[2]);

//    particle.x = distribution_x(generator);
//    particle.y = distribution_y(generator);
//    particle.theta = distribution_theta(generator);

    particle.x = x_f + std_pos[0] * distribution(generator);
    particle.y = y_f + std_pos[1] * distribution(generator);
    particle.theta = theta_f + std_pos[2] * distribution(generator);
    //cin.get();
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

inline double bivariate_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y){
  return exp(-((x-mu_x)*(x-mu_x)/(2.*sig_x*sig_x)+(y-mu_y)*(y-mu_y)/(2.*sig_y*sig_y)))/(2.*M_PI*sig_x*sig_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for(int p = 0; p < num_particles; p++){  // update weight for every particle

    Particle &particle = particles[p];

    // convert observations into map coordinate system
    // rotation + translation

    std::vector<LandmarkObs> observation_in_map_coord;

    for(int j = 0; j < observations.size(); j++){

      LandmarkObs observation = observations[j];

      // if the observations are in sensor range convert to map coordinate system
      if(dist(observations[j].x,observations[j].y,0,0)<=sensor_range){

        LandmarkObs observation_in_map;
        double theta = -particle.theta;
        observation_in_map.x =   observation.x * cos(theta) + observation.y * sin(theta);
        observation_in_map.y = - observation.x * sin(theta) + observation.y * cos(theta);

        observation_in_map.x += particle.x;
        observation_in_map.y += particle.y;

        observation_in_map_coord.push_back(observation_in_map);

      }

    }

    std::vector<LandmarkObs> map_landmarks_observation; // convert Map map_landmarks to LandMarkObs object

    for(int i=0;i<map_landmarks.landmark_list.size();i++){
      LandmarkObs map_observation;
      map_observation.x = map_landmarks.landmark_list[i].x_f;
      map_observation.y = map_landmarks.landmark_list[i].y_f;
      map_observation.id = map_landmarks.landmark_list[i].id_i;
      map_landmarks_observation.push_back(map_observation);
    }


    // calculate the weight for this particle
    double wg = 1.;
    for(int i=0;i<observation_in_map_coord.size();i++){
      // for each observations, find the closest landmark

      double min_dist = 1e100;
      int min_index = -1;

      for(int j=0;j<map_landmarks_observation.size();j++){

        double dst = dist(observation_in_map_coord[i].x,observation_in_map_coord[i].y,map_landmarks_observation[j].x,map_landmarks_observation[j].y);
        if(dst < min_dist){
          min_dist = dst;
          min_index = j;
        }
      }

      // calculate the weight from this observations and its losest landmark.
      wg *= bivariate_normal(observation_in_map_coord[i].x,
                             observation_in_map_coord[i].y,
                             map_landmarks_observation[min_index].x,
                             map_landmarks_observation[min_index].y,
                             std_landmark[0],std_landmark[1]);

    }

    particle.weight = wg;
  } // end loop for particles


  // normalize weights
  double wg_sum = 0.;
  for(int i=0;i<particles.size();i++)
    wg_sum += particles[i].weight;

  for(int i=0;i<particles.size();i++)
    particles[i].weight /= wg_sum;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::default_random_engine generator;

  std::vector<double> weights(num_particles);
  for(int i = 0; i < num_particles; i++){
    weights[i] = particles[i].weight;
  }

  std::discrete_distribution<> sample(weights.begin(), weights.end());

  std::vector<Particle> particles_new(num_particles);

  //for(int i = 0; i < num_particles; i++){
  for(auto &particle : particles_new ){
    particle = particles[sample(generator)];
  }
  particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
