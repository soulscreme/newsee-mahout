package com.soulmedia;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.TrainClassifier;
import org.apache.mahout.classifier.bayes.algorithm.BayesAlgorithm;
import org.apache.mahout.classifier.bayes.common.BayesParameters;
import org.apache.mahout.classifier.bayes.datastore.InMemoryBayesDatastore;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.interfaces.Algorithm;
import org.apache.mahout.classifier.bayes.interfaces.Datastore;
import org.apache.mahout.classifier.bayes.model.ClassifierContext;
import org.apache.mahout.common.nlp.NGrams;

/**
 * Hello world!
 */
public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {
        final BayesParameters params = new BayesParameters();
        params.setGramSize(1);
        params.set("verbose", "true");
        params.set("classifierType", "bayes");
        params.set("defaultCat", "OTHER");
        params.set("encoding", "UTF-8");
        params.set("alpha_i", "1.0");
        params.set("dataSource", "hdfs");
        params.set("basePath", "/tmp/output");

        try {
            Path input = new Path("/tmp/input");
            Path output = new Path("/tmp/output");
            TrainClassifier.trainNaiveBayes(input, output, params);

            Algorithm algorithm = new BayesAlgorithm();
            Datastore datastore = new InMemoryBayesDatastore(params);
            ClassifierContext classifier = new ClassifierContext(algorithm, datastore);
            classifier.initialize();

            final BufferedReader reader = new BufferedReader(new FileReader(args[0]));
            String entry = reader.readLine();
            log.debug("First line: " + entry);

            while (entry != null) {
                log.debug("Processing line: " + entry);

                List<String> document = new NGrams(entry,
                        Integer.parseInt(params.get("gramSize")))
                        .generateNGramsWithoutLabel();

                ClassifierResult result = classifier.classifyDocument(
                        document.toArray(new String[document.size()]),
                        params.get("defaultCat"));

                log.debug("Label: " + result.getLabel() + ", Score: " + result.getScore() + ", " + entry);

                entry = reader.readLine();
            }
        } catch (final IOException ex) {
            ex.printStackTrace();
        } catch (final InvalidDatastoreException ex) {
            ex.printStackTrace();
        }
    }
}
