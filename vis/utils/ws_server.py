import asyncio
import websockets
import time
import json

class WebSocket(object):
    def __init__(self, producer=None, consumer=None, addr='localhost', port=8765):
        self.addr = addr
        self.port = port
        if producer:
            self.producer = producer
        if consumer:
            self.consumer = consumer


    async def producer(self):
        d = {'time': time.time()}
        return json.dumps(d)

    async def consumer(self, message):
        print(f'client connected: {message}')

    async def consumer_handler(self, websocket, path):
        async for message in websocket:
            if asyncio.iscoroutinefunction(self.consumer):
                await self.consumer(message)
            else:
                self.consumer(message)

    async def producer_handler(self, websocket, path):
        while True:
            await asyncio.sleep(0.05)
            if asyncio.iscoroutinefunction(self.producer):  # TODO delete?
                message = await self.producer()
            else:
                message = self.producer()
            await websocket.send(message)

    async def handler(self, websocket, path):
        consumer_task = asyncio.ensure_future(
            self.consumer_handler(websocket, path))
        producer_task = asyncio.ensure_future(
            self.producer_handler(websocket, path))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    def run(self):
        start_server = websockets.serve(self.handler, self.addr, self.port)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    WebSocket().run()
