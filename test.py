def main(args: dict):
    import json

    # 字段配置：字段名与中文表头映射
    field_config = [
        ("orderId", "订单ID"),
        ("orderStatus", "订单状态"),
        ("channel", "下单渠道"),
        ("couponTitle", "项目标题"),
        ("shouldPayMoney", "支付金额")
    ]

    parsed_fields = []

    # 解析字段数据
    for field, _ in field_config:
        value = args.get(field)

        # 空值处理
        if value is None or (isinstance(value, str) and not value.strip()):
            parsed_fields.append(["--"])
            continue

        # 处理逗号分隔字符串
        if isinstance(value, str) and not value.startswith('['):
            str_list = [item.strip() for item in value.split(',') if item.strip()]
            json_list = str_list + ["--"] * max(0, 10 - len(str_list))
        else:
            # 解析JSON
            try:
                json_list = json.loads(value)
                if not isinstance(json_list, list):
                    json_list = [json_list]
            except (json.JSONDecodeError, TypeError):
                json_list = [value]

        # 规范化元素
        processed_list = []
        for item in json_list:
            if item in ({}, None) or (isinstance(item, str) and not item.strip()):
                processed_list.append("--")
            else:
                processed_list.append(str(item).lower() if isinstance(item, bool) else str(item))

        parsed_fields.append(processed_list)

    # 确定记录数（最多10条）
    max_len = max((len(col) for col in parsed_fields), default=0)
    record_count = min(max_len, 10)

    # 构建字典列表
    result_list = []
    for i in range(record_count):
        row_dict = {}
        all_dash = True  # 标记是否全为"--"

        for j, (field, header) in enumerate(field_config):
            # 获取值，若索引超出则用"--"
            if i < len(parsed_fields[j]):
                value = parsed_fields[j][i]
            else:
                value = "--"

            row_dict[header] = value  # 使用中文表头作为字典键
            if value != "--":
                all_dash = False

        # 过滤全"--"的行
        if not all_dash:
            result_list.append(row_dict)

    # 构建符合单引号格式的字符串
    dict_strs = []
    for row_dict in result_list:
        items = []
        for _, header in field_config:  # 按顺序获取表头
            value = row_dict.get(header, "--")  # 使用get方法避免KeyError

            # 构建键值对，键和值都用单引号包裹
            # 对值进行转义处理，确保单引号不会破坏字符串结构
            escaped_value = str(value).replace("'", "\\'")
            items.append(f"'{header}': '{escaped_value}'")

        dict_strs.append("{" + ", ".join(items) + "}")

    # 构建完整的单引号列表
    if dict_strs:
        result_str = "[" + ", ".join(dict_strs) + "]"
    else:
        result_str = "[]"

    # 返回单引号格式的字符串，确保中间确实都是单引号
    return {"OrderCupon": result_str}

