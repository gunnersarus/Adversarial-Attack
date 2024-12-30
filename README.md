# Adversarial-Attack
Adversarial Attack
- Đầu tiên cần tạo một venv trên WSL hoặc Ubuntu để tránh xung đột version giữa các thư viện.
- Tiếp theo tại venv cài các module của 1 trong 2 file requirement_vicuna.txt hoặc requirement_guanaco.txt tương ứng với model muốn thực nghiệm.
- Các bạn có thể chạy từng file riêng lẻ là Vicuna-filter-defense.py hoặc Vicuna-paraphase.py.
- Tuy nhiên nhóm mình đã dựng một flask server tương ứng với hai phương pháp phòng thủ là Filter và Paraphase.
- Để ryn Perplexity-Filter App, mọi người cần tạo một venv và cài các module của model guanaco, và chạy File app.py trong Perplexity-Filter-app.
- Tương tự với Paraphase-app, mọi người cần cài đặt môi trường cho Vicuna, sau đó run app.py trong paraphase-app.
