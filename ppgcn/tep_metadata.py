from __future__ import annotations


XMEAS_INFO = {
    1: ("A feed (Stream 1)", "feed", "kscmh"),
    2: ("D feed (Stream 2)", "feed", "kg/h"),
    3: ("E feed (Stream 3)", "feed", "kg/h"),
    4: ("Total feed (Stream 4)", "feed", "kscmh"),
    5: ("Recycle flow (Stream 8)", "recycle", "kscmh"),
    6: ("Reactor feed rate (Stream 6)", "reactor feed", "kscmh"),
    7: ("Reactor pressure", "reactor", "kPa gauge"),
    8: ("Reactor level", "reactor", "%"),
    9: ("Reactor temperature", "reactor", "deg C"),
    10: ("Purge rate (Stream 9)", "purge", "kscmh"),
    11: ("Product separator temperature", "separator", "deg C"),
    12: ("Product separator level", "separator", "%"),
    13: ("Product separator pressure", "separator", "kPa gauge"),
    14: ("Product separator underflow (Stream 10)", "separator", "m3/h"),
    15: ("Stripper level", "stripper", "%"),
    16: ("Stripper pressure", "stripper", "kPa gauge"),
    17: ("Stripper underflow (Stream 11)", "stripper", "m3/h"),
    18: ("Stripper temperature", "stripper", "deg C"),
    19: ("Stripper steam flow", "stripper", "kg/h"),
    20: ("Compressor work", "compressor", "kW"),
    21: ("Reactor cooling water outlet temperature", "reactor cooling", "deg C"),
    22: ("Separator cooling water outlet temperature", "separator cooling", "deg C"),
    23: ("Component A in reactor feed, Stream 6", "composition stream 6", "mol %"),
    24: ("Component B in reactor feed, Stream 6", "composition stream 6", "mol %"),
    25: ("Component C in reactor feed, Stream 6", "composition stream 6", "mol %"),
    26: ("Component D in reactor feed, Stream 6", "composition stream 6", "mol %"),
    27: ("Component E in reactor feed, Stream 6", "composition stream 6", "mol %"),
    28: ("Component F in reactor feed, Stream 6", "composition stream 6", "mol %"),
    29: ("Component A in purge, Stream 9", "composition stream 9", "mol %"),
    30: ("Component B in purge, Stream 9", "composition stream 9", "mol %"),
    31: ("Component C in purge, Stream 9", "composition stream 9", "mol %"),
    32: ("Component D in purge, Stream 9", "composition stream 9", "mol %"),
    33: ("Component E in purge, Stream 9", "composition stream 9", "mol %"),
    34: ("Component F in purge, Stream 9", "composition stream 9", "mol %"),
    35: ("Component G in purge, Stream 9", "composition stream 9", "mol %"),
    36: ("Component H in purge, Stream 9", "composition stream 9", "mol %"),
    37: ("Component D in stripper product, Stream 11", "composition stream 11", "mol %"),
    38: ("Component E in stripper product, Stream 11", "composition stream 11", "mol %"),
    39: ("Component F in stripper product, Stream 11", "composition stream 11", "mol %"),
    40: ("Component G in stripper product, Stream 11", "composition stream 11", "mol %"),
    41: ("Component H in stripper product, Stream 11", "composition stream 11", "mol %"),
}


XMV_INFO = {
    1: ("D feed flow valve, Stream 2", "feed control", "valve position"),
    2: ("E feed flow valve, Stream 3", "feed control", "valve position"),
    3: ("A feed flow valve, Stream 1", "feed control", "valve position"),
    4: ("Total feed flow valve, Stream 4", "feed control", "valve position"),
    5: ("Compressor recycle valve", "recycle control", "valve position"),
    6: ("Purge valve, Stream 9", "purge control", "valve position"),
    7: ("Separator pot liquid flow valve, Stream 10", "separator control", "valve position"),
    8: ("Stripper liquid product flow valve, Stream 11", "stripper control", "valve position"),
    9: ("Stripper steam valve", "stripper control", "valve position"),
    10: ("Reactor cooling water flow valve", "reactor cooling control", "valve position"),
    11: ("Condenser cooling water flow valve", "separator cooling control", "valve position"),
}


XMEAS_ZH = {
    1: "A 进料流量（物流 1）",
    2: "D 进料流量（物流 2）",
    3: "E 进料流量（物流 3）",
    4: "总进料流量（物流 4）",
    5: "循环物流量（物流 8）",
    6: "反应器进料速率（物流 6）",
    7: "反应器压力",
    8: "反应器液位",
    9: "反应器温度",
    10: "放空流量（物流 9）",
    11: "产品分离器温度",
    12: "产品分离器液位",
    13: "产品分离器压力",
    14: "产品分离器底流（物流 10）",
    15: "汽提塔液位",
    16: "汽提塔压力",
    17: "汽提塔底流（物流 11）",
    18: "汽提塔温度",
    19: "汽提塔蒸汽流量",
    20: "压缩机功率",
    21: "反应器冷却水出口温度",
    22: "分离器冷却水出口温度",
    23: "反应器进料中 A 组分（物流 6）",
    24: "反应器进料中 B 组分（物流 6）",
    25: "反应器进料中 C 组分（物流 6）",
    26: "反应器进料中 D 组分（物流 6）",
    27: "反应器进料中 E 组分（物流 6）",
    28: "反应器进料中 F 组分（物流 6）",
    29: "放空物流中 A 组分（物流 9）",
    30: "放空物流中 B 组分（物流 9）",
    31: "放空物流中 C 组分（物流 9）",
    32: "放空物流中 D 组分（物流 9）",
    33: "放空物流中 E 组分（物流 9）",
    34: "放空物流中 F 组分（物流 9）",
    35: "放空物流中 G 组分（物流 9）",
    36: "放空物流中 H 组分（物流 9）",
    37: "汽提塔产品中 D 组分（物流 11）",
    38: "汽提塔产品中 E 组分（物流 11）",
    39: "汽提塔产品中 F 组分（物流 11）",
    40: "汽提塔产品中 G 组分（物流 11）",
    41: "汽提塔产品中 H 组分（物流 11）",
}


XMV_ZH = {
    1: "D 进料流量阀（物流 2）",
    2: "E 进料流量阀（物流 3）",
    3: "A 进料流量阀（物流 1）",
    4: "总进料流量阀（物流 4）",
    5: "压缩机循环阀",
    6: "放空阀（物流 9）",
    7: "分离器液相流量阀（物流 10）",
    8: "汽提塔液体产品流量阀（物流 11）",
    9: "汽提塔蒸汽阀",
    10: "反应器冷却水流量阀",
    11: "冷凝器冷却水流量阀",
}


AREA_ZH = {
    "feed": "进料系统",
    "feed control": "进料控制",
    "recycle": "循环系统",
    "recycle control": "循环控制",
    "reactor feed": "反应器进料",
    "reactor": "反应器",
    "reactor cooling": "反应器冷却",
    "reactor cooling control": "反应器冷却控制",
    "purge": "放空系统",
    "purge control": "放空控制",
    "separator": "产品分离器",
    "separator control": "分离器控制",
    "separator cooling": "分离器冷却",
    "separator cooling control": "分离器冷却控制",
    "stripper": "汽提塔",
    "stripper control": "汽提塔控制",
    "compressor": "压缩机",
    "composition stream 6": "物流 6 组成",
    "composition stream 9": "物流 9 组成",
    "composition stream 11": "物流 11 组成",
    "unknown": "未知区域",
}


KIND_ZH = {
    "measured": "测量变量",
    "manipulated": "操纵变量",
}


def variable_name(index: int, xmv_start: int = 41) -> str:
    if index >= xmv_start:
        return f"XMV{index - xmv_start + 1}"
    return f"X{index + 1}"


def variable_info(index: int) -> dict[str, str | int]:
    name = variable_name(index)
    if index >= 41:
        number = index - 41 + 1
        description, area, unit = XMV_INFO.get(number, ("Unknown manipulated variable", "unknown", ""))
        description_zh = XMV_ZH.get(number, "未知操纵变量")
        kind = "manipulated"
    else:
        number = index + 1
        description, area, unit = XMEAS_INFO.get(number, ("Unknown measured variable", "unknown", ""))
        description_zh = XMEAS_ZH.get(number, "未知测量变量")
        kind = "measured"
    return {
        "name": name,
        "index": int(index),
        "kind": kind,
        "kind_zh": KIND_ZH.get(kind, kind),
        "description": description,
        "description_zh": description_zh,
        "area": area,
        "area_zh": AREA_ZH.get(area, area),
        "unit": unit,
    }


def edge_physical_meaning(source_index: int, target_index: int) -> str:
    source = variable_info(source_index)
    target = variable_info(target_index)
    if source["area"] == target["area"]:
        return f"same-area coupling in {source['area']}: {source['description']} -> {target['description']}"
    return f"candidate propagation from {source['area']} to {target['area']}: {source['description']} -> {target['description']}"


def edge_physical_meaning_zh(source_index: int, target_index: int) -> str:
    source = variable_info(source_index)
    target = variable_info(target_index)
    if source["area"] == target["area"]:
        return f"同一区域耦合（{source['area_zh']}）：{source['description_zh']} -> {target['description_zh']}"
    return (
        f"候选传播关系：从{source['area_zh']}到{target['area_zh']}，"
        f"{source['description_zh']} -> {target['description_zh']}"
    )
