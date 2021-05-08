package org.ml;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;
public class Csv2Arff {
	public static void main(String[] args) throws Exception{
		//load csv
		CSVLoader loader= new CSVLoader();
		loader.setSource(new File("E:\\A.Sai Preeth\\placements\\oracle\\numeric\\Mnumeric\\carInsurance_train.csv"));
		Instances data = loader.getDataSet();
		
		//save arff
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("E:\\A.Sai Preeth\\placements\\oracle\\numeric\\Mnumeric\\carInsurance_train.arff"));
		saver.writeBatch();
	}

}
