package weka.test;

import com.opencsv.CSVReader;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.Instance;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class Test {
    
    private static final double EPSILON = 0.0000001;
    
	public static void main(String[] args) throws Exception {
    	    ArffLoader arffLoader1 = new ArffLoader();
		arffLoader1.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-winequality-red-test.arff"));
		Instances structure1 = arffLoader1.getStructure();
		ArffLoader arffLoader2 = new ArffLoader();
		arffLoader2.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-winequality-red-test.arff"));
		Instances structure2 = arffLoader2.getStructure();
		ArffLoader loader1 = new ArffLoader();
		loader1.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-winequality-red-train.arff"));
		Instances dataTrain = loader1.getDataSet();
		structure1.setClassIndex(structure1.numAttributes() - 1);
		structure2.setClassIndex(structure2.numAttributes() - 1);
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		
        try (
            Reader reader1 = Files.newBufferedReader(Paths.get("/Users/zhangzikai/Desktop/testOutput.csv"));
            CSVReader csvReader1 = new CSVReader(reader1);
        		Reader reader2 = Files.newBufferedReader(Paths.get("/Users/zhangzikai/Desktop/predByExperts.csv"));
            CSVReader csvReader2 = new CSVReader(reader2);
        		Reader reader3 = Files.newBufferedReader(Paths.get("/Users/zhangzikai/Desktop/testOutput.csv"));
            CSVReader csvReader3 = new CSVReader(reader3);
        ) {
            // Reading records one by one in two String arrays
            String[] weightRecord;
            String[] predByExpert;
            
            String[] weightRecordCmp;
            String[] firstLineExcld;
            
            int numRecord = 0;
            Instance current;
            boolean comparison = true;
            
            while ((current = arffLoader1.getNextInstance(structure1)) != null) {
            	
            		numRecord++;
             
            }
            
            int i = 1;
            
            // exclude the first line for comparison
            firstLineExcld = csvReader3.readNext();
            
            
            // verify the first line
            for (int index = 0; index < firstLineExcld.length; index++) {
	            	if (almostEqual(Double.parseDouble(firstLineExcld[index]), 1.0) == false) {
					comparison = false;
				}
            }
            
            // prepare classifiers for classifying
            TestHelper predByExpertsTest = new TestHelper(dataTrain);
    	    
			
            while (i < numRecord && (current = arffLoader2.getNextInstance(structure2)) != null) {
            	
            		
            		weightRecord = csvReader1.readNext();
            		predByExpert = csvReader2.readNext();
            		// get double array version of weights and predictions for experts
            	    double[] weightsRecord = stringArrayToDouble(weightRecord).clone();
            	    double[] predByExpertsCmp = stringArrayToDouble(predByExpert).clone(); 
            	    
            	    weightRecordCmp = csvReader3.readNext();
            	    // get double array version of weights to be compared with
            	    double[] weightsRecordCmp = stringArrayToDouble(weightRecordCmp).clone();
            	    
            	    // initially, set the temp array to the current array
            	    double[] weightsRecordTemp = weightsRecord.clone();
            	    double[] weightsRecordTemp1 = weightsRecord.clone();
            	    double[] weightsRecordTemp0 = weightsRecord.clone();
            	    
            	    /* verify all predictions by experts */
            	    
                double[] predByExpertsTemp = predByExpertsTest.getPredByExperts(current).clone();
                
                /* verify all the weights except the first record */
                
                // get prediction by ensemble
                double predByEnsemble = predByEnsemble(weightsRecord, predByExpertsCmp);
                
                if (almostEqual(predByEnsemble, 2.0) == false) {
	                // calculate weights of experts for the next record
                		weightsRecordTemp = calWeightsVector(predByEnsemble, current, weightsRecordTemp, weightsRecord, predByExpertsCmp).clone();
                		weightsRecord = stringArrayToDouble(weightRecord).clone();
                		// -Cmp means information stored in file, -Temp means information calculated by this program
                    comparison = comparison && comparison(weightsRecordTemp, weightsRecordCmp) && comparison(predByExpertsTemp, predByExpertsCmp);
                } else {
                		// test
                		System.out.println("haha|" + i + "|" + current.classValue());
                		// end of test
                		
                		//calculate weights of experts for the next record (both predByEnsemble be 0.0 and 1.0)
                		weightsRecordTemp1 = calWeightsVector(1.0, current, weightsRecordTemp1, weightsRecord, predByExpertsCmp).clone();
                		weightsRecordTemp0 = calWeightsVector(0.0, current, weightsRecordTemp0, weightsRecord, predByExpertsCmp).clone();
                		
                		// -Cmp means information stored in file. -Temp means information calculated by this program
                		weightsRecord = stringArrayToDouble(weightRecord).clone();
                    comparison = comparison && (comparison(weightsRecordTemp1, weightsRecordCmp) || comparison(weightsRecordTemp0, weightsRecordCmp)) && comparison(predByExpertsTemp, predByExpertsCmp);
                    
                }
                
                // test
                if (comparison(predByExpertsTemp, predByExpertsCmp) == false) {
                		System.out.println("HaHa|" + i + "|" + current);
                		System.out.println(Arrays.toString(predByExpertsTemp));
                		System.out.println(Arrays.toString(predByExpertsCmp));
                }
                // end of test
                
                i++;
                
            }
            
            System.out.println(comparison);
            
        }
        
    }

	private static double[] calWeightsVector(double predByEnsemble, Instance current, double[] weightsRecordTemp, double[] weightsRecord, double[] predByExpertsCmp) {
		// if the prediction by ensemble model is wrong, set the temp array appropriately for comparison
        if (almostEqual(predByEnsemble, current.classValue()) == false) {
        	
        		for (int index = 0; index < weightsRecord.length; index++) {
    			
        			if (almostEqual(predByExpertsCmp[index], current.classValue()) == false) {
    				
        				weightsRecordTemp[index] = weightsRecord[index] * 0.5;
        				
        			}
        		}
        }
        
        // this can be summarized as a function
        double sum = weightsRecordTemp[0] + weightsRecordTemp[1] + weightsRecordTemp[2] + weightsRecordTemp[3] + 
				weightsRecordTemp[4] + weightsRecordTemp[5] + weightsRecordTemp[6];
        
        for (int index = 0; index < weightsRecordTemp.length; index++) {
        		
        		weightsRecordTemp[index] *= 7.0 / sum;
        	
        }
		return weightsRecordTemp;
	}



	private static boolean comparison(double[] calArray, double[] arrayCmp) {
		
		boolean cmp = true;
		for (int i = 0; i < calArray.length; i++)
		{
			if (almostEqual(calArray[i], arrayCmp[i]) == false) {
				cmp = false;
			}
		}
		return cmp;
	}

	private static boolean almostEqual(double num1, double num2) {
		
		return Math.abs(num1-num2) < EPSILON;
	}

	// method for converting string array to double array (I am not sure if it works to convert exponential term to double)
	private static double[] stringArrayToDouble(String[] stringArray) {
		
		double[] doubleArray = new double[stringArray.length];
		
		for (int i = 0; i < stringArray.length; i++)
		{
			
			doubleArray[i] = Double.parseDouble(stringArray[i]);
		}
		
		return doubleArray;
	}

	// method for returning prediction by ensemble model based on weights and predictions of experts
	private static double predByEnsemble(double[] weightsRecord, double[] predByExperts) {
		
		double weiSum = 0.0;
		
		for(int i = 0; i < weightsRecord.length; i++)
		{
			weiSum += weightsRecord[i] * predByExperts[i];
		}
		
		if (almostEqual(weiSum, 3.5) == true) {
			return 2.0;
		}
		
		double pred = weiSum < 3.5 ?  0.0 : 1.0;
		
		return pred;
	}
}
