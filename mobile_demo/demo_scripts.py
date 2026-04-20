"""
Pre-scripted Vietnamese call scenarios for mobile demo testing.

Each scenario contains:
  - metadata: caller info, call type
  - chunks: list of (speaker, text, delay_ms) tuples
    speaker: "caller" | "user"
    delay_ms: pause before this line appears
"""

SCENARIOS = {
    "normal_friend": {
        "id": "normal_friend",
        "caller_name": "Nguyễn Minh Tuấn",
        "caller_phone": "0912 345 678",
        "caller_avatar": "MT",
        "caller_color": "#4a90d9",
        "call_type": "normal",
        "description": "Cuộc gọi bình thường từ bạn bè",
        "expected_label": "Bình thường",
        "chunks": [
            ("caller", "Alo, Tuấn đây. Tối nay mày rảnh không?", 500),
            ("user",   "Ừ, tao rảnh. Có gì không mày?", 1500),
            ("caller", "Tụi mình rủ nhau đi ăn tối, khoảng sáu giờ rưỡi. Mày tới được không?", 800),
            ("user",   "Ờ được, tao tới được. Ăn ở đâu vậy?", 1500),
            ("caller", "Quán lẩu bên đường Lý Thường Kiệt đó, tụi mày biết chỗ không?", 800),
            ("user",   "Biết rồi, cái quán gần trường đại học đó hả?", 1500),
            ("caller", "Đúng rồi đó. Thôi mày chuẩn bị đi nhe, tao ghé qua đón mày luôn.", 800),
            ("user",   "Ừ okey, cảm ơn mày nhe.", 1500),
            ("caller", "Thôi cúp máy nhe, tao đang lái xe.", 800),
        ],
    },

    "bank_scam": {
        "id": "bank_scam",
        "caller_name": "Nhân viên Ngân hàng",
        "caller_phone": "1800 5555 888",
        "caller_avatar": "NH",
        "caller_color": "#e74c3c",
        "call_type": "scam",
        "description": "Giả danh nhân viên ngân hàng — chiếm đoạt tài sản",
        "expected_label": "Lừa đảo",
        "chunks": [
            ("caller", "Xin chào, tôi gọi từ bộ phận bảo mật Ngân hàng Vietcombank.", 500),
            ("user",   "Dạ, xin hỏi có việc gì ạ?", 1500),
            ("caller", "Chúng tôi phát hiện tài khoản của quý khách đang có dấu hiệu bị tấn công và đã bị tạm khóa.", 800),
            ("user",   "Ôi trời, tài khoản của tôi bị khóa ạ?", 1500),
            ("caller", "Đúng vậy. Để bảo vệ tiền của quý khách, chúng tôi cần quý khách xác minh danh tính ngay bây giờ.", 800),
            ("caller", "Quý khách vui lòng cung cấp số tài khoản và mã OTP gửi về điện thoại để chúng tôi mở khóa.", 1000),
            ("user",   "Nhưng mà cung cấp thông tin qua điện thoại có an toàn không ạ?", 1500),
            ("caller", "Hoàn toàn an toàn. Đây là đường dây chính thức của ngân hàng. Nếu không xác minh trong vòng 30 phút, tài khoản sẽ bị đóng băng vĩnh viễn.", 800),
            ("caller", "Anh/chị vui lòng đọc mã OTP vừa nhận được để chúng tôi xử lý khẩn cấp.", 1000),
            ("user",   "Ờ, để tôi kiểm tra điện thoại...", 1500),
            ("caller", "Nhanh lên ạ, hệ thống chỉ chờ thêm 2 phút nữa thôi. Sau đó tài khoản sẽ bị chặn hoàn toàn.", 800),
        ],
    },

    "police_scam": {
        "id": "police_scam",
        "caller_name": "Cơ quan Công an",
        "caller_phone": "0969 111 222",
        "caller_avatar": "CA",
        "caller_color": "#8e44ad",
        "call_type": "scam",
        "description": "Giả danh công an — đe dọa, tống tiền",
        "expected_label": "Lừa đảo",
        "chunks": [
            ("caller", "Đây là Phòng Cảnh sát hình sự Công an Thành phố Hồ Chí Minh gọi.", 500),
            ("user",   "Dạ, tôi nghe ạ.", 1500),
            ("caller", "Chúng tôi đang điều tra vụ rửa tiền có liên quan đến số tài khoản mang tên anh/chị.", 800),
            ("user",   "Rửa tiền ạ? Tôi không làm gì như vậy hết.", 1500),
            ("caller", "Chúng tôi có bằng chứng. Hiện tại lệnh bắt giữ đã được ký và sẽ được thi hành trong 24 giờ tới.", 800),
            ("caller", "Tuy nhiên anh/chị có thể hợp tác điều tra để minh oan. Anh/chị cần chuyển tiền vào tài khoản phong tỏa do chúng tôi cung cấp.", 1000),
            ("user",   "Chuyển tiền ạ? Tại sao phải chuyển tiền?", 1500),
            ("caller", "Đây là thủ tục tạm giữ tài sản để điều tra, sau khi xong sẽ hoàn trả. Nếu anh/chị không hợp tác, chúng tôi sẽ tiến hành bắt giữ ngay hôm nay.", 800),
            ("caller", "Anh/chị tuyệt đối không được kể chuyện này cho ai khác. Đây là thông tin mật của cuộc điều tra.", 1000),
            ("user",   "Nhưng tôi muốn liên hệ với luật sư trước...", 1500),
            ("caller", "Không được phép. Nếu tiết lộ thông tin, anh/chị sẽ bị khép thêm tội cản trở điều tra và bị bắt ngay lập tức.", 800),
        ],
    },

    "lottery_scam": {
        "id": "lottery_scam",
        "caller_name": "Chương trình Trúng thưởng",
        "caller_phone": "1900 6868",
        "caller_avatar": "TT",
        "caller_color": "#f39c12",
        "call_type": "scam",
        "description": "Giả mạo chương trình trúng thưởng — lừa phí",
        "expected_label": "Lừa đảo",
        "chunks": [
            ("caller", "Xin chúc mừng! Anh/chị vừa trúng giải nhất chương trình khuyến mãi của chúng tôi trị giá 500 triệu đồng!", 500),
            ("user",   "Thật ạ? Tôi có tham gia chương trình nào đâu?", 1500),
            ("caller", "Số điện thoại của anh/chị được hệ thống chọn ngẫu nhiên từ danh sách khách hàng. Đây là giải thưởng thực sự, 100% hợp lệ.", 800),
            ("user",   "Ôi, vậy tôi nhận giải thưởng bằng cách nào?", 1500),
            ("caller", "Rất đơn giản. Anh/chị chỉ cần đóng phí xử lý và thuế giải thưởng là 5 triệu đồng để chúng tôi giải ngân 500 triệu cho anh/chị.", 800),
            ("user",   "Ủa, sao lại phải đóng phí trước vậy?", 1500),
            ("caller", "Đây là quy định pháp luật về thuế giải thưởng. Số tiền này sẽ được khấu trừ vào giải thưởng khi nhận. Nếu không đóng trong hôm nay, giải thưởng sẽ bị hủy.", 800),
            ("caller", "Anh/chị chuyển 5 triệu vào số tài khoản này nhé: MB Bank, chủ tài khoản Nguyễn Văn Thắng.", 1000),
            ("user",   "Năm triệu... nhưng tôi chưa bao giờ tham gia gì cả...", 1500),
            ("caller", "Đây là cơ hội ngàn vàng, anh/chị đừng bỏ lỡ. Chúng tôi có hàng trăm người chờ nhận giải thưởng này. Chuyển ngay đi ạ!", 800),
        ],
    },

    "family_normal": {
        "id": "family_normal",
        "caller_name": "Mẹ",
        "caller_phone": "0905 678 901",
        "caller_avatar": "M",
        "caller_color": "#27ae60",
        "call_type": "normal",
        "description": "Cuộc gọi bình thường từ người thân",
        "expected_label": "Bình thường",
        "chunks": [
            ("caller", "Alo con ơi, mẹ đây. Con đang ở đâu vậy?", 500),
            ("user",   "Dạ mẹ, con đang ở cơ quan ạ. Có gì không mẹ?", 1500),
            ("caller", "À, tối nay con về ăn cơm với gia đình không? Mẹ nấu canh chua con thích đó.", 800),
            ("user",   "Dạ được mẹ. Con về khoảng sáu giờ rưỡi nhé.", 1500),
            ("caller", "Ừ, mẹ chờ con. Nhớ về sớm nghe con, đừng làm thêm giờ nữa.", 800),
            ("user",   "Dạ vâng mẹ. Con sẽ về sớm.", 1500),
            ("caller", "À mà con nhớ ghé chợ mua cho mẹ ít rau muống với đậu hũ nghe. Mẹ quên mua sáng nay rồi.", 800),
            ("user",   "Dạ được mẹ, con nhớ rồi.", 1500),
            ("caller", "Thôi mẹ cúp máy, đang bận nấu ăn đây. Về sớm nghe con.", 800),
        ],
    },
}

def get_all_scenario_ids():
    return list(SCENARIOS.keys())

def get_scenario(scenario_id: str) -> dict:
    return SCENARIOS.get(scenario_id, {})

def get_scenarios_summary() -> list:
    return [
        {
            "id": s["id"],
            "caller_name": s["caller_name"],
            "caller_phone": s["caller_phone"],
            "caller_avatar": s["caller_avatar"],
            "caller_color": s["caller_color"],
            "call_type": s["call_type"],
            "description": s["description"],
            "expected_label": s["expected_label"],
        }
        for s in SCENARIOS.values()
    ]
