"""进程内事件总线：输入主题事件，输出独立订阅队列；依赖标准库 threading 和 queue。"""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import RLock


# 消费者使用对象身份判断终止，避免与业务事件值冲突。
EVENT_BUS_CLOSED = object()


class EventBus:
    """使用有界队列隔离发布者和各个订阅者。"""

    def __init__(self, queue_size: int = 64) -> None:
        """创建采用“满时丢最旧”策略的事件总线。\n\n        Args: queue_size: 每个订阅者队列的最大事件数。\n        Returns: 无。\n        Raises: TypeError: queue_size 不是整数。ValueError: queue_size 小于 1。"""
        if isinstance(queue_size, bool) or not isinstance(queue_size, int):
            raise TypeError("queue_size must be an integer")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        self._queue_size = queue_size
        self._subscribers: dict[str, list[Queue[object]]] = {}
        self._lock = RLock()
        self._closed = False
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        """返回因订阅队列满而丢弃的事件数。\n\n        Args: 无。\n        Returns: 自创建以来累计丢弃数。\n        Raises: 无。"""
        with self._lock:
            return self._dropped_events

    @property
    def closed(self) -> bool:
        """查询总线是否已关闭。\n\n        Args: 无。\n        Returns: 已关闭时为 True。\n        Raises: 无。"""
        with self._lock:
            return self._closed

    def subscribe(self, topic: str) -> Queue[object]:
        """为主题创建一个独立有界订阅队列。\n\n        Args: topic: 非空主题名。\n        Returns: 订阅队列；若已关闭，队列中立即含终止标记。\n        Raises: TypeError: topic 不是字符串。ValueError: topic 为空。"""
        self._validate_topic(topic)
        subscriber: Queue[object] = Queue(maxsize=self._queue_size)
        with self._lock:
            if self._closed:
                subscriber.put_nowait(EVENT_BUS_CLOSED)
            else:
                self._subscribers.setdefault(topic, []).append(subscriber)
        return subscriber

    def unsubscribe(self, topic: str, subscriber: Queue[object]) -> None:
        """移除订阅并向该消费者发送终止标记。\n\n        Args: topic: 订阅时使用的主题。subscriber: subscribe 返回的队列。\n        Returns: 无；重复移除安全。\n        Raises: TypeError: topic 或 subscriber 类型错误。ValueError: topic 为空。"""
        self._validate_topic(topic)
        if not isinstance(subscriber, Queue):
            raise TypeError("subscriber must be a Queue")
        with self._lock:
            subscribers = self._subscribers.get(topic, [])
            if subscriber in subscribers:
                subscribers.remove(subscriber)
                if not subscribers:
                    self._subscribers.pop(topic, None)
                self._signal_closed(subscriber)

    def publish(self, topic: str, event: object) -> None:
        """非阻塞地向主题的全部订阅者发布事件。\n\n        Args: topic: 非空主题名。event: 任意业务事件，但不能是内部终止标记。\n        Returns: 无；无订阅者或总线已关闭时直接返回。\n        Raises: TypeError: topic 类型错误。ValueError: topic 为空或事件是终止标记。"""
        self._validate_topic(topic)
        if event is EVENT_BUS_CLOSED:
            raise ValueError("the close sentinel cannot be published")
        with self._lock:
            if self._closed:
                return
            for subscriber in self._subscribers.get(topic, ()):
                try:
                    subscriber.put_nowait(event)
                except Full:
                    # 每个慢消费者只牺牲自己的最旧事件，不影响其他消费者。
                    try:
                        subscriber.get_nowait()
                        subscriber.task_done()
                    except Empty:
                        pass
                    try:
                        subscriber.put_nowait(event)
                    except Full:
                        pass
                    self._dropped_events += 1

    def close(self) -> None:
        """关闭总线并唤醒全部消费者。\n\n        Args: 无。\n        Returns: 无；可重复调用。\n        Raises: 无。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for subscribers in self._subscribers.values():
                for subscriber in subscribers:
                    self._signal_closed(subscriber)
            self._subscribers.clear()

    @staticmethod
    def _signal_closed(subscriber: Queue[object]) -> None:
        """清空队列并无阻塞地写入终止标记。\n\n        Args: subscriber: 要终止的订阅队列。\n        Returns: 无。\n        Raises: 无。"""
        while True:
            try:
                subscriber.get_nowait()
                subscriber.task_done()
            except Empty:
                break
        subscriber.put_nowait(EVENT_BUS_CLOSED)

    @staticmethod
    def _validate_topic(topic: str) -> None:
        """校验主题名。\n\n        Args: topic: 待校验主题。\n        Returns: 无。\n        Raises: TypeError: topic 不是字符串。ValueError: topic 为空。"""
        if not isinstance(topic, str):
            raise TypeError("topic must be a string")
        if not topic.strip():
            raise ValueError("topic must not be empty")
