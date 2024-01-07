import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class TensorFlowService {

    private Graph graph;
    private Session session;

    public TensorFlowService() {
        // 모델 파일을 로드하고 그래프를 생성합니다.
        try (InputStream is = getClass().getResourceAsStream("/model.pb")) {
            graph = new Graph();
            byte[] graphBytes = IOUtils.toByteArray(is);
            graph.importGraphDef(graphBytes);
            session = new Session(graph);
        } catch (IOException e) {
            throw new RuntimeException("모델 파일을 로드할 수 없습니다.", e);
        }
    }

    public float[] predict(float[] inputData) {
        // 입력 데이터를 사용하여 모델을 실행하고 결과를 반환합니다.
        try (Tensor<Float> input = Tensors.create(inputData)) {
            Tensor<Float> result = session.runner()
                                         .feed("input_tensor_name", input)
                                         .fetch("output_tensor_name")
                                         .run()
                                         .get(0)
                                         .expect(Float.class);
            float[] output = new float[result.numElements()];
            result.copyTo(output);
            return output;
        }
    }
}
