package weka.classifiers.meta;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.lazy.KStar;
import weka.classifiers.trees.*;
import weka.core.*;

public class WeightedMajorityAlgorithm1 extends AbstractClassifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8051760735986874066L;

	protected Classifier[] ensemble = new Classifier[3];

	private double beta = 0.5;
	
	public double[] weight = new double[ensemble.length];
	
	// method for setting the experts
	public void setExperts()
	{
		ensemble[0] = new J48();
		
		ensemble[1] = new BayesNet();
		
		ensemble[2] = new KStar();
	}
	
	// method for building the weighted majority algorithm classifier
	public void buildClassifier(Instances instances) throws Exception
	{
		// build model for three experts on a specific data set
		this.setExperts();
		
		for (int i = 0; i < this.ensemble.length; i++)
		{
			this.ensemble[i].buildClassifier(instances);
		}
				
		// build metaClassifier
		this.buildMetaClassifier(instances);
	
	}

	// method for the main algorithm of building the weighted majority algorithm classifier
	public void buildMetaClassifier(Instances instances) throws Exception
	{
		// number of instances
		int numInstances = instances.numInstances();
		
		// initialize the weight array to [1, 1, 1]
		for (int i = 0; i < ensemble.length; i++)
		{
			weight[i] = 1.0;
		}
		
		// array for storing predictions by Experts
		double[] predByExperts = new double[ensemble.length];
		
		// main loop of the weighted majority algorithm1
		for (int i = 0; i < numInstances; i++)
		{
			// get the array of predictions by experts
			predByExperts = this.getPredByExperts(instances.get(i)); /* I still do not know for sure the function of get() */
					
			// make a weighted sum of these predictions and make a prediction for the ensemble model based on it
			double weiSum = getWeiSum(predByExperts, weight);
			
			double pred = this.predByEnsemble(weiSum, weight);		
			
			// compare the prediction of the ensemble model with the fact
			if (pred != instances.get(i).classValue())
			{
				// if it is wrong, update the weights of the experts who predicted wrong
				for (int index = 0; index < ensemble.length; index++)
				{
					if (predByExperts[index] == pred)
					{
						weight[index] = weight[index] * beta;
					}
				}
				
				double sumWeight = Arrays.stream(weight).sum();
				
				// test
				sumWeight = weight[0] + weight[1] + weight[2];
				// end of test
				
				for (int index = 0; index < ensemble.length; index++)
				{
					weight[index] = 1.0 * 3 * weight[index] / sumWeight;
				}
			}
			
			// test
			this.writeFile("/Users/zhangzikai/Desktop/testOutput.csv", weight);
			// end of test
		}
	}
	
	// method for getting prediction by ensemble model
	public double predByEnsemble(double weiSum, double[] weight) throws Exception
	{
		double sumWeight = Arrays.stream(weight).sum(); //not sure of the function of the method call here
				
		sumWeight = weight[0] + weight[1] + weight[2];
		
		double pred;
		
		if ((sumWeight / 2) > weiSum)
		{
			pred = 0.0;
		}
		else if ((sumWeight / 2) < weiSum)
		{
			pred = 1.0;
		}
		else
		{
			Random rand = new Random();
			
			pred = (double) (rand.nextInt(2));
		}
		
		return pred;
	}
	
	// method for getting prediction by experts
	public double[] getPredByExperts(Instance instance) throws Exception
	{
		double[] predByExperts = new double[3];
		
		double[][] distByExperts = new double[3][2];
		
		for (int i = 0; i < ensemble.length; i++)
		{
			distByExperts[i] = ensemble[i].distributionForInstance(instance); 
		}	                                                                  
		                                                                      	
		for (int j = 0; j < ensemble.length; j++)
		{
			predByExperts[j] = distByExperts[j][0] > distByExperts[j][1] ? 0.0 : 1.0;
		}
		
		return predByExperts;
	}
	
	// method for getting the sum of an array
	public double getWeiSum(double[] newArray, double[] weight)
	{
		double sum = 0.0;
		
		for (int i = 0; i < newArray.length; i++)
		{
			sum += newArray[i] * weight[i];
		}
		
		return sum;
	}
	
	// method for classify an instance
	public double classifyInstance(Instance instance) throws Exception
	{
		// compute the predictions by experts
		double[] predByExperts = new double[ensemble.length];
		
		predByExperts = this.getPredByExperts(instance);
		
		// compute the weighted sum of predictions by experts
		double weiSum = getWeiSum(predByExperts, weight);
		
		// compute the prediction by the ensemble model
		double pred = this.predByEnsemble(weiSum, weight);	
		
		return pred;
	}

    // method for getting the distribution of classes of an instance
	public double[] distributionForInstance(Instance instance) throws Exception {
		//return the distribution of probabilities of classes
		double[] dist = new double[2];
		
		dist[(int) this.classifyInstance(instance)] = 1.0;
		
		return dist;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/* test method for writing double values to file failed
	public void writeFile(String filename, double[] doubleArray) throws IOException
	{
		DataOutputStream testOutput = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename, true)));
		
		for(int i = 0; i < this.ensemble.length - 1; i++)
		{
			testOutput.writeDouble(doubleArray[i]); 
			
			testOutput.writeChar(',');
		}
		
		testOutput.writeDouble(doubleArray[this.ensemble.length - 1]);
		
		testOutput.close();
	}
	*/
	
	// test method
	public void writeFile(String filename, double[] doubleArray) throws IOException
	{
		BufferedWriter testOutput = new BufferedWriter(new FileWriter(filename, true));
		
		testOutput.write(Utils.arrayToString(doubleArray) + '\n');
		
		testOutput.close();
	}
}
