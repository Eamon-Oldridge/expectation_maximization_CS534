import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

      public static void main(String[] args) {

          //handle command line arguments

          if (args.length == 0 || args.length == 1) {
              System.out.println("Insufficient command line arguments.");
              return;
          }

          String filename = args[0];
          int numClusters = Integer.parseInt(args[1]);

        ArrayList<ArrayList<Double>> values = parseFile(filename);
        ExpectationMaximization EM = new ExpectationMaximization(values);
        boolean printLL = false;
        boolean printProb = false;
        EM.start(numClusters, printLL, printProb);

    }

    public static ArrayList<ArrayList<Double>> parseFile(String filename) {

        ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
        BufferedReader fileReader = null;

        try {
            String line = "";
            fileReader = new BufferedReader(new FileReader(filename));

            while ((line = fileReader.readLine()) != null){
                String[] values = line.split(",");
                ArrayList<Double> row = new ArrayList<Double>();
                for (String value: values) {
                    row.add(Double.parseDouble(value));
                }
                list.add(row);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        finally {
            try {
                fileReader.close();
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        return list;
    }

}
