#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 拒识模块客户端
通过 HTTP 调用 Orin NX 上的 Bert Rejector 服务

订阅 ROS2 topic，发布预测结果

用法:
    # 方式1: 作为独立节点运行
    ros2 run your_package rejector_client

    # 方式2: 直接运行
    python client_ros2.py --host 192.168.1.100 --port 8089

    # 方式3: 导入使用
    from client_ros2 import RejectorClient
"""

import argparse
import sys
import os
import time
from typing import Optional

import requests

# ROS2 相关（可选，检测不到时不导入）
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from std_msgs.msg import Bool
    ROS2_AVAILABLE = True
except ImportError:
    pass


# ============== 配置 ==============
DEFAULT_SERVICE_URL = "http://192.168.1.100:8089"  # Orin NX IP
DEFAULT_TIMEOUT = 5.0  # 请求超时秒
# ============== 配置 ==============


class RejectorServiceClient:
    """拒识服务 HTTP 客户端（不依赖 ROS2）"""

    def __init__(self, service_url: str = DEFAULT_SERVICE_URL, timeout: float = DEFAULT_TIMEOUT):
        self.service_url = service_url
        self.timeout = timeout
        self.predict_url = f"{service_url}/predict"
        self.batch_url = f"{service_url}/predict_batch"
        self.health_url = f"{service_url}/"

    def predict(self, text: str) -> Optional[dict]:
        """
        单条预测

        Args:
            text: 输入文本

        Returns:
            {"label": "ready", "confidence": 0.99, "scores": [...], "id": 3, "inference_time_ms": 10.5}
            失败时返回 None
        """
        try:
            resp = requests.post(
                self.predict_url,
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Service call failed: {e}")
            return None

    def predict_batch(self, texts: list) -> Optional[dict]:
        """
        批量预测

        Args:
            texts: 输入文本列表

        Returns:
            {"results": [...], "total_time_ms": 35.5, "avg_time_ms": 11.8}
            失败时返回 None
        """
        try:
            resp = requests.post(
                self.batch_url,
                json={"texts": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Service call failed: {e}")
            return None

    def health_check(self) -> bool:
        """检查服务是否可用"""
        try:
            resp = requests.get(self.health_url, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[INFO] Service healthy: {data}")
                return True
            return False
        except requests.exceptions.RequestException:
            return False


# ============== ROS2 集成 ==============

if ROS2_AVAILABLE:

    class RejectorROS2Node(Node):
        """ROS2 拒识模块节点"""

        def __init__(
            self,
            service_url: str = DEFAULT_SERVICE_URL,
            input_topic: str = "dialog_text",
            output_topic: str = "rejector_result",
        ):
            super().__init__("rejector_client")

            self.client = RejectorServiceClient(service_url=service_url)

            # 订阅输入话题
            self.sub = self.create_subscription(
                String,
                input_topic,
                self._on_text_received,
                10,
            )

            # 发布预测结果
            self.pub = self.create_publisher(String, output_topic, 10)

            # 发布拒识结果（ready/non_ready）
            self.reject_pub = self.create_publisher(Bool, "is_ready", 10)

            self.get_logger().info(f"Rejector Client started, service: {service_url}")
            self.get_logger().info(f"Subscribing to: {input_topic}, Publishing to: {output_topic}")

        def _on_text_received(self, msg: String):
            text = msg.data.strip()
            if not text:
                return

            start = time.perf_counter()
            result = self.client.predict(text)
            elapsed = (time.perf_counter() - start) * 1000

            if result:
                # 发布完整结果
                result_msg = String()
                result_msg.data = f"label={result['label']}, conf={result['confidence']:.4f}, time={elapsed:.1f}ms"
                self.pub.publish(result_msg)

                # 发布 ready 标记（非 ready 内容会被拒识模块过滤）
                is_ready_msg = Bool()
                is_ready_msg.data = (result["label"] == "ready")
                self.reject_pub.publish(is_ready_msg)

                self.get_logger().info(
                    f"[{result['label']}] conf={result['confidence']:.4f} | {text[:50]}"
                )
            else:
                self.get_logger().error(f"Prediction failed for: {text[:50]}")


def main():
    parser = argparse.ArgumentParser(description="ROS2 Rejector Client")
    parser.add_argument("--host", type=str, default="192.168.1.100", help="Orin NX IP")
    parser.add_argument("--port", type=int, default=8089, help="服务端口")
    parser.add_argument("--input-topic", type=str, default="dialog_text", help="输入话题")
    parser.add_argument("--output-topic", type=str, default="rejector_result", help="输出话题")
    args = parser.parse_args()

    service_url = f"http://{args.host}:{args.port}"

    if not ROS2_AVAILABLE:
        print("[WARN] ROS2 not available, running in standalone mode")
        print(f"[INFO] Service URL: {service_url}")

        client = RejectorServiceClient(service_url)

        # 健康检查
        if not client.health_check():
            print("[ERROR] Service not available")
            sys.exit(1)

        # 交互测试
        print("\n输入文本进行测试 (Ctrl+C 退出):\n")
        while True:
            try:
                text = input(">>> ").strip()
                if not text:
                    continue
                result = client.predict(text)
                if result:
                    print(f"  标签: {result['label']}")
                    print(f"  置信度: {result['confidence']:.4f}")
                    print(f"  推理耗时: {result['inference_time_ms']:.2f} ms")
                print()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n退出")
                break
        return

    # ROS2 模式
    rclpy.init(args=sys.argv)
    node = RejectorROS2Node(
        service_url=service_url,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown requested")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
