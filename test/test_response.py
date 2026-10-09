# -*- coding: utf-8 -*-

import contextlib
import gzip
import re
import socket
import ssl
import sys
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy

import mock
import pytest
import six

from urllib3._collections import HTTPHeaderDict
from urllib3.exceptions import (
    DecodeError,
    IncompleteRead,
    InvalidChunkLength,
    InvalidHeader,
    ProtocolError,
    ResponseNotChunked,
    SSLError,
    httplib_IncompleteRead,
)
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import (
    BytesQueueBuffer,
    DeflateDecoder,
    HTTPResponse,
    MultiDecoder,
    brotli,
)
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry

if not hasattr(gzip, "compress"):  # Platform-specific: Python 2.7

    def _gzip_compress(data, compresslevel=9):
        out = BytesIO()
        gzip_s = gzip.GzipFile(fileobj=out, mode="wb", compresslevel=compresslevel)
        gzip_s.write(data)
        gzip_s.close()
        return out.getvalue()

    gzip.compress = _gzip_compress


def deflate2_compress(data):
    compressor = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _limit_decode_calls(response, limit=1000):
    """
    Make ``response`` fail instead of looping forever when decoding never
    finishes, so that a regression fails the test rather than hanging it.
    """
    decode = response._decode
    calls = [0]

    def limited_decode(*args, **kwargs):
        calls[0] += 1
        assert calls[0] <= limit, "Decoding did not finish after %d calls" % limit
        return decode(*args, **kwargs)

    response._decode = limited_decode


if brotli:
    try:
        brotli.Decompressor().process(b"", output_buffer_limit=1024)
        _brotli_gte_1_2_0_available = True
    except (AttributeError, TypeError):
        _brotli_gte_1_2_0_available = False
else:
    _brotli_gte_1_2_0_available = False


class TestBytesQueueBuffer(object):
    def test_single_chunk(self):
        buffer = BytesQueueBuffer()
        assert len(buffer) == 0
        with pytest.raises(RuntimeError, match="buffer is empty"):
            assert buffer.get(10)

        buffer.put(b"foo")
        with pytest.raises(ValueError, match="n should be > 0"):
            buffer.get(-1)

        assert buffer.get(1) == b"f"
        assert buffer.get(2) == b"oo"
        with pytest.raises(RuntimeError, match="buffer is empty"):
            assert buffer.get(10)

    def test_read_too_much(self):
        buffer = BytesQueueBuffer()
        buffer.put(b"foo")
        assert buffer.get(100) == b"foo"

    def test_multiple_chunks(self):
        buffer = BytesQueueBuffer()
        buffer.put(b"foo")
        buffer.put(b"bar")
        buffer.put(b"baz")
        assert len(buffer) == 9

        assert buffer.get(1) == b"f"
        assert len(buffer) == 8
        assert buffer.get(4) == b"ooba"
        assert len(buffer) == 4
        assert buffer.get(4) == b"rbaz"
        assert len(buffer) == 0

    @pytest.mark.skipif(
        sys.version_info < (3, 8), reason="pytest-memray requires Python 3.8+"
    )
    @pytest.mark.limit_memory("12.5 MB")  # assert that we're not doubling memory usage
    def test_memory_usage(self):
        # Allocate 10 1MiB chunks
        buffer = BytesQueueBuffer()
        for i in range(10):
            # This allocates 2MiB, putting the max at around 12MiB. Not sure why.
            buffer.put(bytes(b"\0" * 2 ** 20))

        assert len(buffer.get(10 * 2 ** 20)) == 10 * 2 ** 20

    @pytest.mark.parametrize(
        "get_func",
        (lambda b: b.get(len(b)), lambda b: b.get_all()),
        ids=("get", "get_all"),
    )
    @pytest.mark.limit_memory("10.01 MB", current_thread_only=True)
    def test_memory_usage_single_chunk(self, get_func):
        buffer = BytesQueueBuffer()
        chunk = b"\0" * (10 * 2 ** 20)  # 10 MiB
        buffer.put(chunk)
        assert get_func(buffer) is chunk


# A known random (i.e, not-too-compressible) payload generated with:
#    "".join(random.choice(string.printable) for i in xrange(512))
#    .encode("zlib").encode("base64")
# Randomness in tests == bad, and fixing a seed may not be sufficient.
ZLIB_PAYLOAD = b64decode(
    b"""\
eJwFweuaoQAAANDfineQhiKLUiaiCzvuTEmNNlJGiL5QhnGpZ99z8luQfe1AHoMioB+QSWHQu/L+
lzd7W5CipqYmeVTBjdgSATdg4l4Z2zhikbuF+EKn69Q0DTpdmNJz8S33odfJoVEexw/l2SS9nFdi
pis7KOwXzfSqarSo9uJYgbDGrs1VNnQpT9f8zAorhYCEZronZQF9DuDFfNK3Hecc+WHLnZLQptwk
nufw8S9I43sEwxsT71BiqedHo0QeIrFE01F/4atVFXuJs2yxIOak3bvtXjUKAA6OKnQJ/nNvDGKZ
Khe5TF36JbnKVjdcL1EUNpwrWVfQpFYJ/WWm2b74qNeSZeQv5/xBhRdOmKTJFYgO96PwrHBlsnLn
a3l0LwJsloWpMbzByU5WLbRE6X5INFqjQOtIwYz5BAlhkn+kVqJvWM5vBlfrwP42ifonM5yF4ciJ
auHVks62997mNGOsM7WXNG3P98dBHPo2NhbTvHleL0BI5dus2JY81MUOnK3SGWLH8HeWPa1t5KcW
S5moAj5HexY/g/F8TctpxwsvyZp38dXeLDjSQvEQIkF7XR3YXbeZgKk3V34KGCPOAeeuQDIgyVhV
nP4HF2uWHA=="""
)


@pytest.fixture
def sock():
    s = socket.socket()
    yield s
    s.close()


class TestLegacyResponse(object):
    def test_getheaders(self):
        headers = {"host": "example.com"}
        r = HTTPResponse(headers=headers)
        with pytest.warns(
            DeprecationWarning,
            match=r"HTTPResponse.getheaders\(\) is deprecated",
        ):
            assert r.getheaders() == HTTPHeaderDict(headers)

    def test_getheader(self):
        headers = {"host": "example.com"}
        r = HTTPResponse(headers=headers)
        with pytest.warns(
            DeprecationWarning,
            match=r"HTTPResponse.getheader\(\) is deprecated",
        ):
            assert r.getheader("host") == "example.com"


class TestResponse(object):
    def test_cache_content(self):
        r = HTTPResponse(b"foo")
        assert r._body == b"foo"
        assert r.data == b"foo"
        assert r._body == b"foo"

    def test_cache_content_preload_false(self):
        fp = BytesIO(b"foo")
        r = HTTPResponse(fp, preload_content=False)

        assert not r._body
        assert r.data == b"foo"
        assert r._body == b"foo"
        assert r.data == b"foo"

    def test_default(self):
        r = HTTPResponse()
        assert r.data is None

    def test_none(self):
        r = HTTPResponse(None)
        assert r.data is None

    def test_preload(self):
        fp = BytesIO(b"foo")

        r = HTTPResponse(fp, preload_content=True)

        assert fp.tell() == len(b"foo")
        assert r.data == b"foo"

    def test_no_preload(self):
        fp = BytesIO(b"foo")

        r = HTTPResponse(fp, preload_content=False)

        assert fp.tell() == 0
        assert r.data == b"foo"
        assert fp.tell() == len(b"foo")

    def test_decode_bad_data(self):
        fp = BytesIO(b"\x00" * 10)
        with pytest.raises(DecodeError):
            HTTPResponse(fp, headers={"content-encoding": "deflate"})

    def test_reference_read(self):
        fp = BytesIO(b"foo")
        r = HTTPResponse(fp, preload_content=False)

        assert r.read(1) == b"f"
        assert r.read(2) == b"oo"
        assert r.read() == b""
        assert r.read() == b""

    def test_decode_deflate(self):
        data = zlib.compress(b"foo")

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "deflate"})

        assert r.data == b"foo"

    def test_decode_deflate_case_insensitve(self):
        data = zlib.compress(b"foo")

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "DeFlAtE"})

        assert r.data == b"foo"

    def test_chunked_decoding_deflate(self):
        data = zlib.compress(b"foo")

        fp = BytesIO(data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "deflate"}, preload_content=False
        )

        assert r.read(1) == b"f"
        assert r.read(2) == b"oo"
        assert r.read() == b""
        assert r.read() == b""

    def test_chunked_decoding_deflate2(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()

        fp = BytesIO(data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "deflate"}, preload_content=False
        )

        assert r.read(1) == b"f"
        assert r.read(2) == b"oo"
        assert r.read() == b""
        assert r.read() == b""

    def test_chunked_decoding_gzip(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()

        fp = BytesIO(data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "gzip"}, preload_content=False
        )

        assert r.read(1) == b"f"
        assert r.read(2) == b"oo"
        assert r.read() == b""
        assert r.read() == b""

    def test_decode_gzip_multi_member(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()
        data = data * 3

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "gzip"})

        assert r.data == b"foofoofoo"

    def test_decode_gzip_error(self):
        fp = BytesIO(b"foo")
        with pytest.raises(DecodeError):
            HTTPResponse(fp, headers={"content-encoding": "gzip"})

    def test_decode_gzip_swallow_garbage(self):
        # When data comes from multiple calls to read(), data after
        # the first zlib error (here triggered by garbage) should be
        # ignored.
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()
        data = data * 3 + b"foo"

        fp = BytesIO(data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "gzip"}, preload_content=False
        )
        ret = b""
        for _ in range(100):
            ret += r.read(1)
            if r.closed:
                break

        assert ret == b"foofoofoo"

    def test_chunked_decoding_gzip_swallow_garbage(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()
        data = data * 3 + b"foo"

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "gzip"})

        assert r.data == b"foofoofoo"

    @onlyBrotlipy()
    def test_decode_brotli(self):
        data = brotli.compress(b"foo")

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "br"})
        assert r.data == b"foo"

    @onlyBrotlipy()
    def test_chunked_decoding_brotli(self):
        data = brotli.compress(b"foobarbaz")

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "br"}, preload_content=False)

        ret = b""
        for _ in range(100):
            ret += r.read(1)
            if r.closed:
                break
        assert ret == b"foobarbaz"

    @onlyBrotlipy()
    def test_decode_brotli_error(self):
        fp = BytesIO(b"foo")
        with pytest.raises(DecodeError):
            HTTPResponse(fp, headers={"content-encoding": "br"})

    _test_compressor_params = [
        ("deflate1", ("deflate", zlib.compress)),
        ("deflate2", ("deflate", deflate2_compress)),
        ("gzip", ("gzip", gzip.compress)),
    ]
    if _brotli_gte_1_2_0_available:
        _test_compressor_params.append(("brotli", ("br", brotli.compress)))
    else:
        _test_compressor_params.append(("brotli", None))

    @pytest.mark.parametrize("read_method", ("read",))
    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_compressor_params],
        ids=[d[0] for d in _test_compressor_params],
    )
    def test_read_with_all_data_already_in_decompressor(
        self,
        request,
        read_method,
        data,
    ):
        if data is None:
            pytest.skip(
                "Proper %s decoder is not available" % (request.node.callspec.id,)
            )
        original_data = b"bar" * 1000
        name, compress_func = data
        compressed_data = compress_func(original_data)
        fp = mock.Mock(read=mock.Mock(return_value=b""))
        r = HTTPResponse(fp, headers={"content-encoding": name}, preload_content=False)
        # Put all data in the decompressor's buffer.
        r._init_decoder()
        assert r._decoder is not None
        decoded = r._decoder.decompress(compressed_data, max_length=0)
        if name == "br":
            # It's known that some Brotli libraries do not respect
            # `max_length`.
            r._decoded_buffer.put(decoded)
        else:
            assert decoded == b""
        # Read the data via `HTTPResponse`.
        read = getattr(r, read_method)
        assert read(0) == b""
        assert read(2500) == original_data[:2500]
        assert read(500) == original_data[2500:]
        assert read(0) == b""
        assert read() == b""

    @pytest.mark.parametrize("read_method", ("read_chunked", "stream"))
    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_compressor_params],
        ids=[d[0] for d in _test_compressor_params],
    )
    def test_read_chunked_with_all_data_already_in_decompressor(
        self,
        request,
        read_method,
        data,
    ):
        if data is None:
            pytest.skip(
                "Proper %s decoder is not available" % (request.node.callspec.id,)
            )
        original_data = b"0" * 100000  # 100 KB
        name, compress_func = data
        compressed_data = compress_func(original_data)
        httplib_r = httplib.HTTPResponse(MockSock)
        httplib_r.fp = MockChunkedEncodingResponse([compressed_data])
        r = HTTPResponse(
            httplib_r,
            preload_content=False,
            headers={"transfer-encoding": "chunked", "content-encoding": name},
        )
        # Mark the current chunk as fully read.
        r.chunk_left = 0
        # Feed all compressed data to the decoder.
        r._init_decoder()
        assert r._decoder is not None
        initially_decoded = r._decoder.decompress(compressed_data, max_length=0)
        result = list(getattr(r, read_method)(amt=10240, decode_content=True))
        if name == "br":
            # It's known that some Brotli libraries do not respect
            # `max_length`.
            result = [initially_decoded] + list(result)
        else:
            assert initially_decoded == b""
            assert all(len(chunk) == 10240 for chunk in result[:-1])
            assert len(result[-1]) == len(original_data) % 10240
        assert b"".join(result) == original_data

    @pytest.mark.parametrize(
        "delta",
        (
            0,  # First read from socket returns all compressed data.
            -1,  # First read from socket returns all but one byte of compressed data.
        ),
    )
    @pytest.mark.parametrize("read_method", ("read",))
    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_compressor_params],
        ids=[d[0] for d in _test_compressor_params],
    )
    def test_decode_with_max_length_close_to_compressed_data_size(
        self,
        request,
        delta,
        read_method,
        data,
    ):
        """
        Test decoding when the first read from the socket returns all or
        almost all the compressed data, but then it has to be
        decompressed in a couple of read calls.
        """
        if data is None:
            pytest.skip(
                "Proper %s decoder is not available" % (request.node.callspec.id,)
            )

        original_data = b"foo" * 1000
        name, compress_func = data
        compressed_data = compress_func(original_data)
        fp = BytesIO(compressed_data)
        r = HTTPResponse(fp, headers={"content-encoding": name}, preload_content=False)
        initial_limit = len(compressed_data) + delta
        read = getattr(r, read_method)
        initial_chunk = read(amt=initial_limit, decode_content=True)
        assert len(initial_chunk) == initial_limit
        assert (
            len(read(amt=len(original_data), decode_content=True))
            == len(original_data) - initial_limit
        )

    # Prepare 50 MB of compressed data outside of the test measuring
    # memory usage.
    _test_memory_usage_decode_with_max_length_params = [
        (
            params[0],
            (params[1][0], params[1][1](b"A" * (50 * 2 ** 20))) if params[1] else None,
        )
        for params in _test_compressor_params
    ]

    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_memory_usage_decode_with_max_length_params],
        ids=[d[0] for d in _test_memory_usage_decode_with_max_length_params],
    )
    @pytest.mark.parametrize("read_method", ("read", "read_chunked", "stream"))
    # Decoders consume different amounts of memory during decompression.
    # We set the 10 MB limit to ensure that the whole decompressed data
    # is not stored unnecessarily.
    #
    # FYI, the following consumption was observed for the test with
    # `read` on CPython 3.14.0:
    #   - deflate: 2.3 MiB
    #   - deflate2: 2.1 MiB
    #   - gzip: 2.1 MiB
    #   - brotli:
    #     - brotli v1.2.0: 9 MiB
    #     - brotlicffi v1.2.0.0: 6 MiB
    #     - brotlipy v0.7.0: 105.8 MiB
    #   - zstd: 4.5 MiB
    @pytest.mark.limit_memory("10 MB", current_thread_only=True)
    def test_memory_usage_decode_with_max_length(
        self,
        request,
        read_method,
        data,
    ):
        if data is None:
            pytest.skip(
                "Proper %s decoder is not available" % (request.node.callspec.id,)
            )

        name, compressed_data = data
        limit1 = 1024 * 1024  # 1 MiB
        # We test with two read calls because the second call may be
        # able to use the internal buffer filled by the first call, and
        # we want to ensure that full decompression is never triggered
        # by the second call. The limit for the second call is lowered
        # to make sure that the internal buffer is used for the Brotli
        # case specifically https://github.com/google/brotli/issues/1396
        limit2 = 1024  # 1 KiB
        if read_method in ("read_chunked", "stream"):
            httplib_r = httplib.HTTPResponse(MockSock)
            httplib_r.chunked = True
            httplib_r.chunk_left = 1
            httplib_r.fp = MockChunkedEncodingResponse([compressed_data])
            r = HTTPResponse(
                httplib_r,
                preload_content=False,
                headers={"transfer-encoding": "chunked", "content-encoding": name},
            )
            for limit in (limit1, limit2):
                next(getattr(r, read_method)(amt=limit, decode_content=True))
        else:
            fp = BytesIO(compressed_data)
            r = HTTPResponse(
                fp, headers={"content-encoding": name}, preload_content=False
            )
            for limit in (limit1, limit2):
                getattr(r, read_method)(amt=limit, decode_content=True)

        # Check that the internal decoded buffer is empty unless brotli
        # is used.
        # Google's brotli library does not fully respect the output
        # buffer limit: https://github.com/google/brotli/issues/1396
        # And unmaintained brotlipy cannot limit the output buffer size.
        if name != "br" or brotli.__name__ == "brotlicffi":
            assert len(r._decoded_buffer) == 0

        # Check that memory usage is still within the limit while the
        # connection is being drained, meaning that the call does not
        # decompress the whole content.
        r.drain_conn()
        assert r._decoder is None
        assert len(r._decoded_buffer) == 0

    # A decompression bomb: ~8 MiB of zeroes behind a few kilobytes on the
    # wire. Only the zlib-based decoders are exercised here because they are
    # always available and because they honor the output limit exactly.
    # Brotli is intentionally left out of the strict per-chunk assertion:
    # Google's brotli library does not fully respect the output buffer limit
    # (https://github.com/google/brotli/issues/1396), and brotlipy cannot
    # limit it at all. Brotli is covered by the tests above instead.
    _test_bomb_params = [
        (
            params[0],
            (params[1][0], params[1][1](b"\0" * (8 * 2 ** 20))),
        )
        for params in _test_compressor_params
        if params[1] is not None and params[1][0] != "br"
    ]

    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_bomb_params],
        ids=[d[0] for d in _test_bomb_params],
    )
    @pytest.mark.parametrize(
        "read_method", ("read", "stream", "chunked_stream", "read_chunked")
    )
    def test_decompression_bomb_is_bounded_by_amt(self, read_method, data):
        """
        A highly compressible body must never be expanded past the requested
        ``amt`` by a single read/iteration, no matter how small ``amt`` is
        compared to the compression ratio.
        """
        name, compressed_data = data
        expected_length = 8 * 2 ** 20
        amt = 1024

        if read_method in ("read", "stream"):
            r = HTTPResponse(
                BytesIO(compressed_data),
                headers={"content-encoding": name},
                preload_content=False,
            )
            if read_method == "stream":
                iterator = r.stream(amt=amt, decode_content=True)
            else:

                def chunks():
                    while True:
                        chunk = r.read(amt, decode_content=True)
                        if not chunk:
                            return
                        yield chunk

                iterator = chunks()
        else:
            httplib_r = httplib.HTTPResponse(MockSock)
            httplib_r.fp = MockChunkedEncodingResponse([compressed_data])
            r = HTTPResponse(
                httplib_r,
                preload_content=False,
                headers={"transfer-encoding": "chunked", "content-encoding": name},
            )
            if read_method == "chunked_stream":
                iterator = r.stream(amt=amt, decode_content=True)
            else:
                iterator = r.read_chunked(amt=amt, decode_content=True)

        total = 0
        count = 0
        for chunk in iterator:
            # This is the vulnerability: before the fix a single call decoded
            # everything the compressed input expanded to, ignoring `amt`.
            assert len(chunk) <= amt
            total += len(chunk)
            count += 1
        assert total == expected_length
        assert count == expected_length // amt

    def test_drain_conn_does_not_decode_content(self):
        """
        Draining a connection, e.g. a redirect response body that the caller
        never reads, must not decompress the body. A malicious server could
        otherwise amplify a few kilobytes on the wire into a huge allocation
        without any read limit being applied.
        """
        compressed_data = gzip.compress(b"\0" * (8 * 2 ** 20))
        fp = BytesIO(compressed_data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "gzip"}, preload_content=False
        )

        with mock.patch("urllib3.response.GzipDecoder.decompress") as decompress:
            r.drain_conn()

        # This is the vulnerability: before the fix the whole 8 MiB were
        # decompressed just for the result to be thrown away.
        decompress.assert_not_called()
        assert r._has_decoded_content is False
        # The body is still read off the wire so the connection can be reused.
        assert fp.tell() == len(compressed_data)

    def test_drain_conn_does_not_decode_after_partial_read(self):
        """
        CVE-2026-44432: draining after a partial decoded read must not decode
        the remainder either. `_has_decoded_content` being True is not a
        licence to decompress everything that is left in a single operation.
        """
        compressed_data = gzip.compress(b"\0" * (8 * 2 ** 20))
        fp = BytesIO(compressed_data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "gzip"}, preload_content=False
        )

        assert r.read(1) == b"\0"
        assert r._has_decoded_content is True

        with mock.patch("urllib3.response.GzipDecoder.decompress") as decompress:
            r.drain_conn()

        decompress.assert_not_called()
        # The decoder and its buffer are dropped so nothing is retained.
        assert r._decoder is None
        assert len(r._decoded_buffer) == 0
        # The body is still read off the wire so the connection can be reused.
        assert fp.tell() == len(compressed_data)

    @pytest.mark.parametrize(
        "compress_func",
        (zlib.compress, deflate2_compress),
        ids=("deflate1", "deflate2"),
    )
    def test_deflate_decoder_stops_at_end_of_stream(self, compress_func):
        original_data = b"A" * 100
        # The encoded body contains bytes after the end of the Deflate stream.
        compressed_data = compress_func(original_data) + b"tail"
        decoder = DeflateDecoder()

        decoded = decoder.decompress(compressed_data, max_length=50)
        assert decoded == original_data[:50]
        assert decoder.has_unconsumed_tail
        decoded += decoder.decompress(b"", max_length=50)
        assert decoded == original_data

        # The end of the Deflate stream has been reached, so the trailing
        # bytes must not be reported as data that is still to be decoded.
        # Otherwise callers draining the decoder until it has no unconsumed
        # tail (e.g. `HTTPResponse.read_chunked`) loop forever.
        assert not decoder.has_unconsumed_tail
        # Input received after the end of the stream is not accepted.
        unused_data = decoder._obj.unused_data
        assert decoder.decompress(b"more", max_length=50) == b""
        assert decoder.decompress(b"more") == b""
        assert decoder._obj.unused_data == unused_data
        assert not decoder.has_unconsumed_tail

    @pytest.mark.parametrize("read_method", ("stream", "read_chunked"))
    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_compressor_params],
        ids=[d[0] for d in _test_compressor_params],
    )
    def test_read_chunked_with_trailing_data_does_not_hang(
        self, request, read_method, data
    ):
        if data is None:
            pytest.skip(
                "Proper %s decoder is not available" % (request.node.callspec.id,)
            )
        original_data = b"A" * 100
        content_encoding, compress_func = data
        if content_encoding == "br" and brotli.__name__ == "brotlicffi":
            pytest.skip("This case is not supported by brotlicffi")
        compressed_data = compress_func(original_data) + b"tail"
        httplib_r = httplib.HTTPResponse(MockSock)
        httplib_r.fp = MockChunkedEncodingResponse([compressed_data])
        r = HTTPResponse(
            httplib_r,
            preload_content=False,
            headers={
                "transfer-encoding": "chunked",
                "content-encoding": content_encoding,
            },
        )
        _limit_decode_calls(r)
        stream = getattr(r, read_method)(len(original_data) // 2, decode_content=True)
        if content_encoding == "br":
            with pytest.raises(DecodeError):
                list(stream)
        else:
            assert b"".join(stream) == original_data

    _test_trailing_data_params = [
        params for params in _test_compressor_params if params[0] != "brotli"
    ]

    @pytest.mark.parametrize(
        "data",
        [d[1] for d in _test_trailing_data_params],
        ids=[d[0] for d in _test_trailing_data_params],
    )
    def test_stream_with_trailing_data_does_not_hang(self, data):
        original_data = b"A" * 100
        content_encoding, compress_func = data
        compressed_data = compress_func(original_data) + b"tail"
        r = HTTPResponse(
            BytesIO(compressed_data),
            preload_content=False,
            headers={"content-encoding": content_encoding},
        )
        _limit_decode_calls(r)
        stream = r.stream(len(original_data) // 2, decode_content=True)
        assert b"".join(stream) == original_data

    def test_multi_decoding_deflate_deflate(self):
        data = zlib.compress(zlib.compress(b"foo"))

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "deflate, deflate"})

        assert r.data == b"foo"

    def test_multi_decoding_deflate_gzip(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(zlib.compress(b"foo"))
        data += compress.flush()

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "deflate, gzip"})

        assert r.data == b"foo"

    def test_multi_decoding_gzip_gzip(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()

        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(data)
        data += compress.flush()

        fp = BytesIO(data)
        r = HTTPResponse(fp, headers={"content-encoding": "gzip, gzip"})

        assert r.data == b"foo"

    def test_read_multi_decoding_too_many_links(self):
        fp = BytesIO(b"foo")
        with pytest.raises(
            DecodeError, match="Too many content encodings in the chain: 6 > 5"
        ):
            HTTPResponse(
                fp,
                headers={
                    "content-encoding": "gzip, deflate, gzip, deflate, gzip, deflate"
                },
            )

    def test_read_multi_decoding_deflate_deflate(self):
        msg = b"foobarbaz" * 42
        data = zlib.compress(zlib.compress(msg))

        fp = BytesIO(data)
        r = HTTPResponse(
            fp, headers={"content-encoding": "deflate, deflate"}, preload_content=False
        )

        assert r.read(3) == b"foo"
        assert r.read(3) == b"bar"
        assert r.read(3) == b"baz"
        assert r.read(9) == b"foobarbaz"
        assert r.read(9 * 3) == b"foobarbaz" * 3
        assert r.read(9 * 37) == b"foobarbaz" * 37
        assert r.read() == b""

    def test_multi_decoding_max_links(self):
        # A chain at the limit is still built, ...
        assert len(MultiDecoder("gzip, deflate, gzip, deflate, gzip")._decoders) == 5

        # ... but one more link is refused before any decoder is created.
        with pytest.raises(
            DecodeError, match="Too many content encodings in the chain: 6 > 5"
        ):
            MultiDecoder("gzip, deflate, gzip, deflate, gzip, deflate")

        # A chain long enough to be used for resource amplification is
        # refused as well.
        with pytest.raises(
            DecodeError, match="Too many content encodings in the chain: 1000 > 5"
        ):
            MultiDecoder(", ".join(["gzip"] * 1000))

    def test_body_blob(self):
        resp = HTTPResponse(b"foo")
        assert resp.data == b"foo"
        assert resp.closed

    def test_io(self, sock):
        fp = BytesIO(b"foo")
        resp = HTTPResponse(fp, preload_content=False)

        assert not resp.closed
        assert resp.readable()
        assert not resp.writable()
        with pytest.raises(IOError):
            resp.fileno()

        resp.close()
        assert resp.closed

        # Try closing with an `httplib.HTTPResponse`, because it has an
        # `isclosed` method.
        try:
            hlr = httplib.HTTPResponse(sock)
            resp2 = HTTPResponse(hlr, preload_content=False)
            assert not resp2.closed
            resp2.close()
            assert resp2.closed
        finally:
            hlr.close()

        # also try when only data is present.
        resp3 = HTTPResponse("foodata")
        with pytest.raises(IOError):
            resp3.fileno()

        resp3._fp = 2
        # A corner case where _fp is present but doesn't have `closed`,
        # `isclosed`, or `fileno`.  Unlikely, but possible.
        assert resp3.closed
        with pytest.raises(IOError):
            resp3.fileno()

    def test_io_closed_consistently(self, sock):
        try:
            hlr = httplib.HTTPResponse(sock)
            hlr.fp = BytesIO(b"foo")
            hlr.chunked = 0
            hlr.length = 3
            with HTTPResponse(hlr, preload_content=False) as resp:
                assert not resp.closed
                assert not resp._fp.isclosed()
                assert not is_fp_closed(resp._fp)
                assert not resp.isclosed()
                resp.read()
                assert resp.closed
                assert resp._fp.isclosed()
                assert is_fp_closed(resp._fp)
                assert resp.isclosed()
        finally:
            hlr.close()

    def test_io_bufferedreader(self):
        fp = BytesIO(b"foo")
        resp = HTTPResponse(fp, preload_content=False)
        br = BufferedReader(resp)

        assert br.read() == b"foo"

        br.close()
        assert resp.closed

        # HTTPResponse.read() by default closes the response
        # https://github.com/urllib3/urllib3/issues/1305
        fp = BytesIO(b"hello\nworld")
        resp = HTTPResponse(fp, preload_content=False)
        with pytest.raises(ValueError) as ctx:
            list(BufferedReader(resp))
        assert str(ctx.value) == "readline of closed file"

        b = b"fooandahalf"
        fp = BytesIO(b)
        resp = HTTPResponse(fp, preload_content=False)
        br = BufferedReader(resp, 5)

        br.read(1)  # sets up the buffer, reading 5
        assert len(fp.read()) == (len(b) - 5)

        # This is necessary to make sure the "no bytes left" part of `readinto`
        # gets tested.
        while not br.closed:
            br.read(5)

    def test_io_not_autoclose_bufferedreader(self):
        fp = BytesIO(b"hello\nworld")
        resp = HTTPResponse(fp, preload_content=False, auto_close=False)
        reader = BufferedReader(resp)
        assert list(reader) == [b"hello\n", b"world"]

        assert not reader.closed
        assert not resp.closed
        with pytest.raises(StopIteration):
            next(reader)

        reader.close()
        assert reader.closed
        assert resp.closed
        with pytest.raises(ValueError) as ctx:
            next(reader)
        assert str(ctx.value) == "readline of closed file"

    def test_io_textiowrapper(self):
        fp = BytesIO(b"\xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f")
        resp = HTTPResponse(fp, preload_content=False)
        br = TextIOWrapper(resp, encoding="utf8")

        assert br.read() == u"äöüß"

        br.close()
        assert resp.closed

        # HTTPResponse.read() by default closes the response
        # https://github.com/urllib3/urllib3/issues/1305
        fp = BytesIO(
            b"\xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f\n\xce\xb1\xce\xb2\xce\xb3\xce\xb4"
        )
        resp = HTTPResponse(fp, preload_content=False)
        with pytest.raises(ValueError) as ctx:
            if six.PY2:
                # py2's implementation of TextIOWrapper requires `read1`
                # method which is provided by `BufferedReader` wrapper
                resp = BufferedReader(resp)
            list(TextIOWrapper(resp))
        assert re.match("I/O operation on closed file.?", str(ctx.value))

    def test_io_not_autoclose_textiowrapper(self):
        fp = BytesIO(
            b"\xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f\n\xce\xb1\xce\xb2\xce\xb3\xce\xb4"
        )
        resp = HTTPResponse(fp, preload_content=False, auto_close=False)
        if six.PY2:
            # py2's implementation of TextIOWrapper requires `read1`
            # method which is provided by `BufferedReader` wrapper
            resp = BufferedReader(resp)
        reader = TextIOWrapper(resp, encoding="utf8")
        assert list(reader) == [u"äöüß\n", u"αβγδ"]

        assert not reader.closed
        assert not resp.closed
        with pytest.raises(StopIteration):
            next(reader)

        reader.close()
        assert reader.closed
        assert resp.closed
        with pytest.raises(ValueError) as ctx:
            next(reader)
        assert re.match("I/O operation on closed file.?", str(ctx.value))

    def test_streaming(self):
        fp = BytesIO(b"foo")
        resp = HTTPResponse(fp, preload_content=False)
        stream = resp.stream(2, decode_content=False)

        assert next(stream) == b"fo"
        assert next(stream) == b"o"
        with pytest.raises(StopIteration):
            next(stream)

    def test_streaming_tell(self):
        fp = BytesIO(b"foo")
        resp = HTTPResponse(fp, preload_content=False)
        stream = resp.stream(2, decode_content=False)

        position = 0

        position += len(next(stream))
        assert 2 == position
        assert position == resp.tell()

        position += len(next(stream))
        assert 3 == position
        assert position == resp.tell()

        with pytest.raises(StopIteration):
            next(stream)

    def test_gzipped_streaming(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()

        fp = BytesIO(data)
        resp = HTTPResponse(
            fp, headers={"content-encoding": "gzip"}, preload_content=False
        )
        stream = resp.stream(2)

        assert next(stream) == b"fo"
        assert next(stream) == b"o"
        with pytest.raises(StopIteration):
            next(stream)

    def test_gzipped_streaming_tell(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        uncompressed_data = b"foo"
        data = compress.compress(uncompressed_data)
        data += compress.flush()

        fp = BytesIO(data)
        resp = HTTPResponse(
            fp, headers={"content-encoding": "gzip"}, preload_content=False
        )
        stream = resp.stream()

        # Read everything
        payload = next(stream)
        assert payload == uncompressed_data

        assert len(data) == resp.tell()

        with pytest.raises(StopIteration):
            next(stream)

    def test_deflate_streaming_tell_intermediate_point(self):
        # Ensure that ``tell()`` returns the correct number of bytes when
        # part-way through streaming compressed content.
        NUMBER_OF_READS = 10
        PART_SIZE = 64

        class MockCompressedDataReading(BytesIO):
            """
            A BytesIO-like reader returning ``payload`` in ``NUMBER_OF_READS``
            calls to ``read``.
            """

            def __init__(self, payload, payload_part_size):
                self.payloads = [
                    payload[i * payload_part_size : (i + 1) * payload_part_size]
                    for i in range(NUMBER_OF_READS + 1)
                ]

                assert b"".join(self.payloads) == payload

            def read(self, _):
                # Amount is unused.
                if len(self.payloads) > 0:
                    return self.payloads.pop(0)
                return b""

        uncompressed_data = zlib.decompress(ZLIB_PAYLOAD)

        payload_part_size = len(ZLIB_PAYLOAD) // NUMBER_OF_READS
        fp = MockCompressedDataReading(ZLIB_PAYLOAD, payload_part_size)
        resp = HTTPResponse(
            fp, headers={"content-encoding": "deflate"}, preload_content=False
        )
        stream = resp.stream(PART_SIZE)

        parts_positions = [(part, resp.tell()) for part in stream]
        end_of_stream = resp.tell()

        with pytest.raises(StopIteration):
            next(stream)

        parts, positions = zip(*parts_positions)

        # Check that the payload is equal to the uncompressed data
        payload = b"".join(parts)
        assert uncompressed_data == payload

        # Check that the positions in the stream are correct
        # It is difficult to determine programatically what the positions
        # returned by `tell` will be because the `HTTPResponse.read` method may
        # call socket `read` a couple of times if it doesn't have enough data
        # in the buffer or not call socket `read` at all if it has enough. All
        # this depends on the message, how it was compressed, what is
        # `PART_SIZE` and `payload_part_size`.
        # So for simplicity the expected values are hardcoded.
        expected = (92, 184, 230, 276, 322, 368, 414, 460)
        assert expected == positions

        # Check that the end of the stream is in the correct place
        assert len(ZLIB_PAYLOAD) == end_of_stream

        # Check that all parts have expected length
        expected_last_part_size = len(uncompressed_data) % PART_SIZE
        whole_parts = len(uncompressed_data) // PART_SIZE
        if expected_last_part_size == 0:
            expected_lengths = [PART_SIZE] * whole_parts
        else:
            expected_lengths = [PART_SIZE] * whole_parts + [expected_last_part_size]
        assert expected_lengths == [len(part) for part in parts]

    def test_deflate_streaming(self):
        data = zlib.compress(b"foo")

        fp = BytesIO(data)
        resp = HTTPResponse(
            fp, headers={"content-encoding": "deflate"}, preload_content=False
        )
        stream = resp.stream(2)

        assert next(stream) == b"fo"
        assert next(stream) == b"o"
        with pytest.raises(StopIteration):
            next(stream)

    def test_deflate2_streaming(self):
        compress = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
        data = compress.compress(b"foo")
        data += compress.flush()

        fp = BytesIO(data)
        resp = HTTPResponse(
            fp, headers={"content-encoding": "deflate"}, preload_content=False
        )
        stream = resp.stream(2)

        assert next(stream) == b"fo"
        assert next(stream) == b"o"
        with pytest.raises(StopIteration):
            next(stream)

    def test_empty_stream(self):
        fp = BytesIO(b"")
        resp = HTTPResponse(fp, preload_content=False)
        stream = resp.stream(2, decode_content=False)

        with pytest.raises(StopIteration):
            next(stream)

    @pytest.mark.parametrize(
        "preload_content, amt",
        [(True, None), (False, None), (False, 10 * 2 ** 20)],
    )
    @pytest.mark.limit_memory("25 MB")
    def test_buffer_memory_usage_decode_one_chunk(self, preload_content, amt):
        content_length = 10 * 2 ** 20  # 10 MiB
        fp = BytesIO(zlib.compress(b"\0" * content_length))
        resp = HTTPResponse(
            fp,
            preload_content=preload_content,
            headers={"content-encoding": "deflate"},
        )
        data = resp.data if preload_content else resp.read(amt)
        assert len(data) == content_length

    @pytest.mark.parametrize(
        "preload_content, amt",
        [(True, None), (False, None), (False, 10 * 2 ** 20)],
    )
    @pytest.mark.limit_memory("10.5 MB")
    def test_buffer_memory_usage_no_decoding(self, preload_content, amt):
        content_length = 10 * 2 ** 20  # 10 MiB
        fp = BytesIO(b"\0" * content_length)
        resp = HTTPResponse(fp, preload_content=preload_content, decode_content=False)
        data = resp.data if preload_content else resp.read(amt)
        assert len(data) == content_length

    def test_length_no_header(self):
        fp = BytesIO(b"12345")
        resp = HTTPResponse(fp, preload_content=False)
        assert resp.length_remaining is None

    def test_length_w_valid_header(self):
        headers = {"content-length": "5"}
        fp = BytesIO(b"12345")

        resp = HTTPResponse(fp, headers=headers, preload_content=False)
        assert resp.length_remaining == 5

    def test_length_w_bad_header(self):
        garbage = {"content-length": "foo"}
        fp = BytesIO(b"12345")

        resp = HTTPResponse(fp, headers=garbage, preload_content=False)
        assert resp.length_remaining is None

        garbage["content-length"] = "-10"
        resp = HTTPResponse(fp, headers=garbage, preload_content=False)
        assert resp.length_remaining is None

    def test_length_when_chunked(self):
        # This is expressly forbidden in RFC 7230 sec 3.3.2
        # We fall back to chunked in this case and try to
        # handle response ignoring content length.
        headers = {"content-length": "5", "transfer-encoding": "chunked"}
        fp = BytesIO(b"12345")

        resp = HTTPResponse(fp, headers=headers, preload_content=False)
        assert resp.length_remaining is None

    def test_length_with_multiple_content_lengths(self):
        headers = {"content-length": "5, 5, 5"}
        garbage = {"content-length": "5, 42"}
        fp = BytesIO(b"abcde")

        resp = HTTPResponse(fp, headers=headers, preload_content=False)
        assert resp.length_remaining == 5

        with pytest.raises(InvalidHeader):
            HTTPResponse(fp, headers=garbage, preload_content=False)

    def test_length_after_read(self):
        headers = {"content-length": "5"}

        # Test no defined length
        fp = BytesIO(b"12345")
        resp = HTTPResponse(fp, preload_content=False)
        resp.read()
        assert resp.length_remaining is None

        # Test our update from content-length
        fp = BytesIO(b"12345")
        resp = HTTPResponse(fp, headers=headers, preload_content=False)
        resp.read()
        assert resp.length_remaining == 0

        # Test partial read
        fp = BytesIO(b"12345")
        resp = HTTPResponse(fp, headers=headers, preload_content=False)
        data = resp.stream(2)
        next(data)
        assert resp.length_remaining == 3

    def test_mock_httpresponse_stream(self):
        # Mock out a HTTP Request that does enough to make it through urllib3's
        # read() and close() calls, and also exhausts and underlying file
        # object.
        class MockHTTPRequest(object):
            self.fp = None

            def read(self, amt):
                data = self.fp.read(amt)
                if not data:
                    self.fp = None

                return data

            def close(self):
                self.fp = None

        bio = BytesIO(b"foo")
        fp = MockHTTPRequest()
        fp.fp = bio
        resp = HTTPResponse(fp, preload_content=False)
        stream = resp.stream(2)

        assert next(stream) == b"fo"
        assert next(stream) == b"o"
        with pytest.raises(StopIteration):
            next(stream)

    def test_mock_transfer_encoding_chunked(self):
        stream = [b"fo", b"o", b"bar"]
        fp = MockChunkedEncodingResponse(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )

        for i, c in enumerate(resp.stream()):
            assert c == stream[i]

    def test_mock_gzipped_transfer_encoding_chunked_decoded(self):
        """Show that we can decode the gzipped and chunked body."""

        def stream():
            # Set up a generator to chunk the gzipped body
            compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
            data = compress.compress(b"foobar")
            data += compress.flush()
            for i in range(0, len(data), 2):
                yield data[i : i + 2]

        fp = MockChunkedEncodingResponse(list(stream()))
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        headers = {"transfer-encoding": "chunked", "content-encoding": "gzip"}
        resp = HTTPResponse(r, preload_content=False, headers=headers)

        data = b""
        for c in resp.stream(decode_content=True):
            data += c

        assert b"foobar" == data

    def test_mock_transfer_encoding_chunked_custom_read(self):
        stream = [b"foooo", b"bbbbaaaaar"]
        fp = MockChunkedEncodingResponse(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )
        expected_response = [b"fo", b"oo", b"o", b"bb", b"bb", b"aa", b"aa", b"ar"]
        response = list(resp.read_chunked(2))
        assert expected_response == response

    def test_mock_transfer_encoding_chunked_unlmtd_read(self):
        stream = [b"foooo", b"bbbbaaaaar"]
        fp = MockChunkedEncodingResponse(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )
        assert stream == list(resp.read_chunked())

    def test_read_not_chunked_response_as_chunks(self):
        fp = BytesIO(b"foo")
        resp = HTTPResponse(fp, preload_content=False)
        r = resp.read_chunked()
        with pytest.raises(ResponseNotChunked):
            next(r)

    def test_buggy_incomplete_read(self):
        # Simulate buggy versions of Python (<2.7.4)
        # See http://bugs.python.org/issue16298
        content_length = 1337
        fp = BytesIO(b"")
        resp = HTTPResponse(
            fp,
            headers={"content-length": str(content_length)},
            preload_content=False,
            enforce_content_length=True,
        )
        with pytest.raises(ProtocolError) as ctx:
            resp.read(3)

        orig_ex = ctx.value.args[1]
        assert isinstance(orig_ex, IncompleteRead)
        assert orig_ex.partial == 0
        assert orig_ex.expected == content_length

    def test_incomplete_chunk(self):
        stream = [b"foooo", b"bbbbaaaaar"]
        fp = MockChunkedIncompleteRead(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )
        with pytest.raises(ProtocolError) as ctx:
            next(resp.read_chunked())

        orig_ex = ctx.value.args[1]
        assert isinstance(orig_ex, httplib_IncompleteRead)

    def test_invalid_chunk_length(self):
        stream = [b"foooo", b"bbbbaaaaar"]
        fp = MockChunkedInvalidChunkLength(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )
        with pytest.raises(ProtocolError) as ctx:
            next(resp.read_chunked())

        orig_ex = ctx.value.args[1]
        assert isinstance(orig_ex, InvalidChunkLength)
        assert orig_ex.length == six.b(fp.BAD_LENGTH_LINE)

    def test_chunked_response_without_crlf_on_end(self):
        stream = [b"foo", b"bar", b"baz"]
        fp = MockChunkedEncodingWithoutCRLFOnEnd(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )
        assert stream == list(resp.stream())

    def test_chunked_response_with_extensions(self):
        stream = [b"foo", b"bar"]
        fp = MockChunkedEncodingWithExtensions(stream)
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            r, preload_content=False, headers={"transfer-encoding": "chunked"}
        )
        assert stream == list(resp.stream())

    def test_chunked_head_response(self):
        r = httplib.HTTPResponse(MockSock, method="HEAD")
        r.chunked = True
        r.chunk_left = None
        resp = HTTPResponse(
            "",
            preload_content=False,
            headers={"transfer-encoding": "chunked"},
            original_response=r,
        )
        assert resp.chunked is True

        resp.supports_chunked_reads = lambda: True
        resp.release_conn = mock.Mock()
        for _ in resp.stream():
            continue
        resp.release_conn.assert_called_once_with()

    def test_get_case_insensitive_headers(self):
        headers = {"host": "example.com"}
        r = HTTPResponse(headers=headers)
        assert r.headers.get("host") == "example.com"
        assert r.headers.get("Host") == "example.com"

    def test_retries(self):
        fp = BytesIO(b"")
        resp = HTTPResponse(fp)
        assert resp.retries is None
        retry = Retry()
        resp = HTTPResponse(fp, retries=retry)
        assert resp.retries == retry

    def test_geturl(self):
        fp = BytesIO(b"")
        request_url = "https://example.com"
        resp = HTTPResponse(fp, request_url=request_url)
        assert resp.geturl() == request_url

    def test_geturl_retries(self):
        fp = BytesIO(b"")
        resp = HTTPResponse(fp, request_url="http://example.com")
        request_histories = [
            RequestHistory(
                method="GET",
                url="http://example.com",
                error=None,
                status=301,
                redirect_location="https://example.com/",
            ),
            RequestHistory(
                method="GET",
                url="https://example.com/",
                error=None,
                status=301,
                redirect_location="https://www.example.com",
            ),
        ]
        retry = Retry(history=request_histories)
        resp = HTTPResponse(fp, retries=retry)
        assert resp.geturl() == "https://www.example.com"

    @pytest.mark.parametrize(
        ["payload", "expected_stream"],
        [
            (b"", []),
            (b"\n", [b"\n"]),
            (b"\n\n\n", [b"\n", b"\n", b"\n"]),
            (b"abc\ndef", [b"abc\n", b"def"]),
            (b"Hello\nworld\n\n\n!", [b"Hello\n", b"world\n", b"\n", b"\n", b"!"]),
        ],
    )
    def test__iter__(self, payload, expected_stream):
        actual_stream = []
        for chunk in HTTPResponse(BytesIO(payload), preload_content=False):
            actual_stream.append(chunk)

        assert actual_stream == expected_stream

    def test__iter__decode_content(self):
        def stream():
            # Set up a generator to chunk the gzipped body
            compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
            data = compress.compress(b"foo\nbar")
            data += compress.flush()
            for i in range(0, len(data), 2):
                yield data[i : i + 2]

        fp = MockChunkedEncodingResponse(list(stream()))
        r = httplib.HTTPResponse(MockSock)
        r.fp = fp
        headers = {"transfer-encoding": "chunked", "content-encoding": "gzip"}
        resp = HTTPResponse(r, preload_content=False, headers=headers)

        data = b""
        for c in resp:
            data += c

        assert b"foo\nbar" == data

    def test_non_timeout_ssl_error_on_read(self):
        mac_error = ssl.SSLError(
            "SSL routines", "ssl3_get_record", "decryption failed or bad record mac"
        )

        @contextlib.contextmanager
        def make_bad_mac_fp():
            fp = BytesIO(b"")
            with mock.patch.object(fp, "read") as fp_read:
                # mac/decryption error
                fp_read.side_effect = mac_error
                yield fp

        with make_bad_mac_fp() as fp:
            with pytest.raises(SSLError) as e:
                HTTPResponse(fp)
            assert e.value.args[0] == mac_error

        with make_bad_mac_fp() as fp:
            resp = HTTPResponse(fp, preload_content=False)
            with pytest.raises(SSLError) as e:
                resp.read()
            assert e.value.args[0] == mac_error


class MockChunkedEncodingResponse(object):
    def __init__(self, content):
        """
        content: collection of str, each str is a chunk in response
        """
        self.content = content
        self.index = 0  # This class iterates over self.content.
        self.closed = False
        self.cur_chunk = b""
        self.chunks_exhausted = False

    @staticmethod
    def _encode_chunk(chunk):
        # In the general case, we can't decode the chunk to unicode
        length = "%X\r\n" % len(chunk)
        return length.encode() + chunk + b"\r\n"

    def _pop_new_chunk(self):
        if self.chunks_exhausted:
            return b""
        try:
            chunk = self.content[self.index]
        except IndexError:
            chunk = b""
            self.chunks_exhausted = True
        else:
            self.index += 1
        chunk = self._encode_chunk(chunk)
        if not isinstance(chunk, bytes):
            chunk = chunk.encode()
        return chunk

    def pop_current_chunk(self, amt=-1, till_crlf=False):
        if amt > 0 and till_crlf:
            raise ValueError("Can't specify amt and till_crlf.")
        if len(self.cur_chunk) <= 0:
            self.cur_chunk = self._pop_new_chunk()
        if till_crlf:
            try:
                i = self.cur_chunk.index(b"\r\n")
            except ValueError:
                # No CRLF in current chunk -- probably caused by encoder.
                self.cur_chunk = b""
                return b""
            else:
                chunk_part = self.cur_chunk[: i + 2]
                self.cur_chunk = self.cur_chunk[i + 2 :]
                return chunk_part
        elif amt <= -1:
            chunk_part = self.cur_chunk
            self.cur_chunk = b""
            return chunk_part
        else:
            try:
                chunk_part = self.cur_chunk[:amt]
            except IndexError:
                chunk_part = self.cur_chunk
                self.cur_chunk = b""
            else:
                self.cur_chunk = self.cur_chunk[amt:]
            return chunk_part

    def readline(self, limit=-1):
        return self.pop_current_chunk(till_crlf=True)

    def read(self, amt=-1):
        return self.pop_current_chunk(amt)

    def flush(self):
        # Python 3 wants this method.
        pass

    def close(self):
        self.closed = True


class MockChunkedIncompleteRead(MockChunkedEncodingResponse):
    def _encode_chunk(self, chunk):
        return "9999\r\n%s\r\n" % chunk.decode()


class MockChunkedInvalidChunkLength(MockChunkedEncodingResponse):
    BAD_LENGTH_LINE = "ZZZ\r\n"

    def _encode_chunk(self, chunk):
        return "%s%s\r\n" % (self.BAD_LENGTH_LINE, chunk.decode())


class MockChunkedEncodingWithoutCRLFOnEnd(MockChunkedEncodingResponse):
    def _encode_chunk(self, chunk):
        return "%X\r\n%s%s" % (
            len(chunk),
            chunk.decode(),
            "\r\n" if len(chunk) > 0 else "",
        )


class MockChunkedEncodingWithExtensions(MockChunkedEncodingResponse):
    def _encode_chunk(self, chunk):
        return "%X;asd=qwe\r\n%s\r\n" % (len(chunk), chunk.decode())


class MockSock(object):
    @classmethod
    def makefile(cls, *args, **kwargs):
        return
