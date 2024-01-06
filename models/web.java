<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
<dependency>
    <groupId>com.squareup.retrofit2</groupId>
    <artifactId>retrofit</artifactId>
    <version>2.9.0</version>
</dependency>
<dependency>
    <groupId>com.squareup.okhttp3</groupId>
    <artifactId>okhttp</artifactId>
    <version>4.9.0</version>
</dependency>

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import java.util.HashMap;
import java.util.Map;

@RestController
public class RecommendationController {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String tensorflowServingUrl = "http://localhost:8501/v1/models/groupbuy_recommendation:predict";

    @GetMapping("/recommend/{userId}")
    public String getRecommendations(@PathVariable int userId) {
        Map<String, Object> data = new HashMap<>();
        data.put("instances", new int[][]{{userId}});

        String response = restTemplate.postForObject(tensorflowServingUrl, data, String.class);

        return response;
    }
}
