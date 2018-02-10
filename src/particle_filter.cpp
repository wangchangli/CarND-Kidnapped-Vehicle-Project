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

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	 

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std[0]);
	
	// TODO: Create normal distributions for y and theta
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	
	for (int i = 0; i < num_particles; ++i) {
		 Particle p;
		 p.id = i;
		 p.x = dist_x(gen);
		 p.y = dist_y(gen);
		 p.theta= dist_theta(gen);	 
		 p.weight = 1.0;

		 particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(0, std_pos[0]);
	
	// TODO: Create normal distributions for y and theta
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i) {
	   if (fabs(yaw_rate) < 0.00001) {  
	      particles[i].x += velocity * delta_t * cos(particles[i].theta);
	      particles[i].y += velocity * delta_t * sin(particles[i].theta);
	    } else {
	      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
	      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
	      particles[i].theta += yaw_rate * delta_t;
	    }

	    particles[i].x += dist_x(gen);
	    particles[i].y += dist_y(gen);
	    particles[i].theta += dist_theta(gen);	    
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {

		LandmarkObs observation = observations[i];

		double min_dist = numeric_limits<double>::max();

		int closest_predicte_id = -1;

		for (int j = 0; j < predicted.size(); j++) {
		  LandmarkObs predicte = predicted[j];
		  
		  double cur_dist = dist(observation.x, observation.y, predicte.x, predicte.y);

		  if (cur_dist < min_dist) {
		    min_dist = cur_dist;
		    closest_predicte_id = predicte.id;
		  }
	  	}

	  	observations[i].id = closest_predicte_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
  for (int i = 0; i < num_particles; i++) {

    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector<LandmarkObs> predictions;

    // convert map landmarks to LandmarkObs
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // trans observation measurement from car's coordinate system to map coordinate system
	// ref https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/5c50790c-5370-4c80-aff6-334659d5c0d9/concepts/57e4e914-ccc8-4a69-be9d-d14cc2c9889f
	vector<LandmarkObs> transformed_obs;
    for (int j = 0; j < observations.size(); j++) {
      double x_map = p_x + cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y;
      double y_map = p_y + sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y;
      transformed_obs.push_back(LandmarkObs{ observations[j].id, x_map, y_map });
    }

    dataAssociation(predictions, transformed_obs);

    // init weight
    particles[i].weight = 1.0;

	// ref https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/5c50790c-5370-4c80-aff6-334659d5c0d9/concepts/0a756b5c-458b-491f-b560-ac18b251f14d
    for (int j = 0; j < transformed_obs.size(); j++) {

      double tobs_x = transformed_obs[j].x;
      double tobs_y = transformed_obs[j].y;
      double mu_x, mu_y;

      int associated_prediction = transformed_obs[j].id;

      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          mu_x = predictions[k].x;
          mu_y = predictions[k].y;
          break;
        }
      }

      // calculate the particle weight
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];

      double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
      double exponent= pow((tobs_x - mu_x), 2) / (2 * pow(sig_x,2)) + pow((tobs_y - mu_y), 2)/(2 * pow(sig_y,2));

      // calculate weight using normalization terms and exponent
      double weight= gauss_norm * exp(-exponent);

      particles[i].weight *= weight;
    }
  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	vector<Particle> new_particles(num_particles);
	for ( int i = 0; i < num_particles; i++ ) {
		discrete_distribution<int> index(weights.begin(), weights.end());
		new_particles[i] = particles[index(gen)];
	}	

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
