import contextlib
import json
import os.path
import shutil
import socket
import ssl
import sys
import tempfile
import warnings
from test import (
    LONG_TIMEOUT,
    SHORT_TIMEOUT,
    onlyPy2,
    onlyPy3,
    onlySecureTransport,
    withPyOpenSSL,
)

import pytest
import trustme

from dummyserver.server import DEFAULT_CA, HAS_IPV6, get_unreachable_address
from dummyserver.testcase import HTTPDummyProxyTestCase, IPv6HTTPDummyProxyTestCase
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import VerifiedHTTPSConnection, connection_from_url
from urllib3.exceptions import (
    ConnectTimeoutError,
    InsecureRequestWarning,
    MaxRetryError,
    ProxyError,
    ProxySchemeUnknown,
    ProxySchemeUnsupported,
    ReadTimeoutError,
    SSLError,
    SubjectAltNameWarning,
)
from urllib3.poolmanager import ProxyManager, proxy_from_url
from urllib3.util import Timeout
from urllib3.util.retry import RequestHistory
from urllib3.util.ssl_ import create_urllib3_context

from .. import TARPIT_HOST, requires_network

# Retry failed tests
pytestmark = pytest.mark.flaky


def assert_is_verified(pm, proxy, target):
    pool = list(pm.pools._container.values())[-1]  # retrieve last pool entry
    connection = (
        pool.pool.queue[-1] if pool.pool is not None else None
    )  # retrieve last connection entry

    assert connection is not None
    assert connection.proxy_is_verified is proxy
    assert connection.is_verified is target


def create_stdlib_context(cert_reqs=ssl.CERT_REQUIRED):
    """
    Return a stdlib ``ssl.SSLContext`` configured like
    ``create_urllib3_context(cert_reqs=cert_reqs)``.

    ``create_urllib3_context()`` returns a pyOpenSSL-backed context while
    pyOpenSSL is injected into urllib3 (which can be left over by a skipped
    ``withPyOpenSSL`` test on Python 2), whereas the tests using this helper
    inspect the stdlib ``ssl.SSLSocket`` wrapping the proxy connection.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = cert_reqs
    return context


class TestHTTPProxyManager(HTTPDummyProxyTestCase):
    @classmethod
    def setup_class(cls):
        super(TestHTTPProxyManager, cls).setup_class()
        cls.http_url = "http://%s:%d" % (cls.http_host, cls.http_port)
        cls.http_url_alt = "http://%s:%d" % (cls.http_host_alt, cls.http_port)
        cls.https_url = "https://%s:%d" % (cls.https_host, cls.https_port)
        cls.https_url_alt = "https://%s:%d" % (cls.https_host_alt, cls.https_port)
        cls.proxy_url = "http://%s:%d" % (cls.proxy_host, cls.proxy_port)
        cls.https_proxy_url = "https://%s:%d" % (
            cls.proxy_host,
            cls.https_proxy_port,
        )

        # Generate another CA to test verification failure
        cls.certs_dir = tempfile.mkdtemp()
        bad_ca = trustme.CA()

        cls.bad_ca_path = os.path.join(cls.certs_dir, "ca_bad.pem")
        bad_ca.cert_pem.write_to_path(cls.bad_ca_path)

    @classmethod
    def teardown_class(cls):
        super(TestHTTPProxyManager, cls).teardown_class()
        shutil.rmtree(cls.certs_dir)

    def test_basic_proxy(self):
        with proxy_from_url(self.proxy_url, ca_certs=DEFAULT_CA) as http:
            r = http.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            r = http.request("GET", "%s/" % self.https_url)
            assert r.status == 200

    @onlyPy3
    def test_https_proxy(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            r = https.request("GET", "%s/" % self.https_url)
            assert r.status == 200

            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200

    @onlyPy3
    def test_https_proxy_with_proxy_ssl_context(self):
        proxy_ssl_context = create_urllib3_context()
        proxy_ssl_context.load_verify_locations(DEFAULT_CA)
        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ssl_context,
            ca_certs=DEFAULT_CA,
        ) as https:
            r = https.request("GET", "%s/" % self.https_url)
            assert r.status == 200

            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200

    @withPyOpenSSL
    def test_https_proxy_with_proxy_ssl_context_pyopenssl(self):
        proxy_ssl_context = create_urllib3_context()
        proxy_ssl_context.load_verify_locations(DEFAULT_CA)

        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ssl_context,
        ) as https:
            response = https.request("GET", "%s/" % self.http_url)
            assert response.status == 200

    def test_is_verified_https_proxy_to_http_target(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200
            assert_is_verified(https, proxy=True, target=False)

    def test_is_verified_https_proxy_to_http_target_without_proxy_config(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            https.connection_pool_kw["_proxy_config"] = None
            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200
            assert_is_verified(https, proxy=True, target=False)

    @onlyPy3
    def test_is_verified_https_proxy_to_https_target(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            r = https.request("GET", "%s/" % self.https_url)
            assert r.status == 200
            assert_is_verified(https, proxy=True, target=True)

    def test_is_verified_https_proxy_forwarding_to_https_target(self):
        with proxy_from_url(
            self.https_proxy_url,
            ca_certs=DEFAULT_CA,
            use_forwarding_for_https=True,
        ) as https:
            r = https.request("GET", "%s/" % self.https_url)
            assert r.status == 200
            assert_is_verified(https, proxy=True, target=False)

    @pytest.mark.parametrize("target_scheme", ["http", "https"])
    def test_https_proxy_forwarding_verified_proxy_no_insecure_warning(
        self, target_scheme
    ):
        target_url = self.https_url if target_scheme == "https" else self.http_url
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with proxy_from_url(
                self.https_proxy_url,
                ca_certs=DEFAULT_CA,
                use_forwarding_for_https=True,
            ) as https:
                r = https.request("GET", "%s/" % target_url)
                assert r.status == 200

        assert [x for x in w if issubclass(x.category, InsecureRequestWarning)] == []

    def test_https_proxy_forwarding_unverified_proxy_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with proxy_from_url(
                self.https_proxy_url,
                cert_reqs="NONE",
                use_forwarding_for_https=True,
            ) as https:
                r = https.request("GET", "%s/" % self.https_url)
                assert r.status == 200

        messages = [
            str(x.message) for x in w if issubclass(x.category, InsecureRequestWarning)
        ]
        assert [
            msg
            for msg in messages
            if "Unverified HTTPS connection done to an HTTPS proxy." in msg
        ]
        assert [
            msg for msg in messages if "Unverified HTTPS request is being made" in msg
        ]

    @onlyPy2
    def test_https_proxy_not_supported(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            with pytest.raises(ProxySchemeUnsupported) as excinfo:
                https.request("GET", "%s/" % self.https_url)

            assert "is not supported in Python 2" in str(excinfo.value)

    @withPyOpenSSL
    @onlyPy3
    def test_https_proxy_pyopenssl_not_supported(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            with pytest.raises(ProxySchemeUnsupported) as excinfo:
                https.request("GET", "%s/" % self.https_url)

            assert "isn't available on non-native SSLContext" in str(excinfo.value)

    @onlySecureTransport
    @onlyPy3
    def test_https_proxy_securetransport_not_supported(self):
        with proxy_from_url(self.https_proxy_url, ca_certs=DEFAULT_CA) as https:
            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            with pytest.raises(ProxySchemeUnsupported) as excinfo:
                https.request("GET", "%s/" % self.https_url)

            assert "isn't available on non-native SSLContext" in str(excinfo.value)

    def test_https_proxy_forwarding_for_https(self):
        with proxy_from_url(
            self.https_proxy_url,
            ca_certs=DEFAULT_CA,
            use_forwarding_for_https=True,
        ) as https:
            r = https.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            r = https.request("GET", "%s/" % self.https_url)
            assert r.status == 200

    @requires_network
    @pytest.mark.parametrize(
        "proxy_ssl_context_kw",
        [
            ("ssl_context",),
            ("proxy_ssl_context",),
            ("ssl_context", "proxy_ssl_context"),
        ],
    )
    def test_https_proxy_forwarding_for_https_with_custom_context(
        self, proxy_ssl_context_kw
    ):
        """
        Test that an HTTP request succeeds when using a forwarding HTTPS
        proxy with an SSL context provided via either
        ``proxy_ssl_context``, ``ssl_context`` (fallback), or both.
        """

        proxy_ctx = ssl.create_default_context()
        proxy_ctx.load_verify_locations(DEFAULT_CA)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            proxy = proxy_from_url(
                self.https_proxy_url,
                use_forwarding_for_https=True,
                **{kw: proxy_ctx for kw in proxy_ssl_context_kw}
            )
        future_warnings = [
            x
            for x in w
            if issubclass(x.category, FutureWarning) and "ssl_context" in str(x.message)
        ]
        if "ssl_context" in proxy_ssl_context_kw:
            assert len(future_warnings) == 1
        else:
            assert future_warnings == []

        with proxy:
            resp = proxy.request("GET", self.https_url)
            assert resp.status == 200

    @requires_network
    def test_forwarding_proxy_tls_error_with_fallback(self):
        """
        Test that when only ``ssl_context`` is passed to a forwarding
        HTTPS proxy, a ``FutureWarning`` is emitted and if the TLS
        handshake with the proxy fails using that context, the error is
        a TLS error.
        """
        ctx = ssl.create_default_context()

        with pytest.warns(FutureWarning, match="ssl_context"):
            proxy = proxy_from_url(
                self.https_proxy_url,
                ssl_context=ctx,
                use_forwarding_for_https=True,
            )

        with proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request("GET", self.https_url)
            assert type(e.value.reason) == SSLError

    @requires_network
    def test_forwarding_proxy_ssl_context_precedence(self):
        """
        Test that when both ``ssl_context`` and ``proxy_ssl_context``
        are passed to a forwarding HTTPS proxy, ``proxy_ssl_context`` is
        used for the proxy connection and ``ssl_context`` is ignored
        (with a warning).
        """
        ssl_context = create_stdlib_context(cert_reqs=ssl.CERT_REQUIRED)
        proxy_ssl_context = create_stdlib_context(cert_reqs=ssl.CERT_NONE)

        with pytest.warns(FutureWarning, match="ssl_context"):
            proxy = proxy_from_url(
                self.https_proxy_url,
                proxy_ssl_context=proxy_ssl_context,
                ssl_context=ssl_context,
                use_forwarding_for_https=True,
            )

        with proxy:
            pool = proxy.connection_from_url(self.https_url)
            with contextlib.closing(pool._new_conn()) as conn:
                conn.connect()
                assert isinstance(conn.sock, ssl.SSLSocket)
                assert conn.sock.context is proxy_ssl_context
                assert proxy_ssl_context.verify_mode == ssl.CERT_NONE

    @requires_network
    def test_forwarding_proxy_ssl_context_fallback(self):
        """
        Test that when only ``ssl_context`` is passed to a forwarding
        HTTPS proxy, a ``FutureWarning`` is emitted and ``ssl_context``
        is used as the proxy TLS context (fallback).
        """
        ssl_context = create_stdlib_context(cert_reqs=ssl.CERT_NONE)

        with pytest.warns(FutureWarning, match="ssl_context"):
            proxy = proxy_from_url(
                self.https_proxy_url,
                ssl_context=ssl_context,
                cert_reqs=ssl.CERT_REQUIRED,
                ca_certs=DEFAULT_CA,
                use_forwarding_for_https=True,
            )

        assert proxy.proxy_ssl_context is ssl_context
        assert proxy.proxy_config is not None
        assert proxy.proxy_config.ssl_context is None

        with proxy:
            pool = proxy.connection_from_url(self.https_url)
            with contextlib.closing(pool._new_conn()) as conn:
                conn.connect()
                assert isinstance(conn.sock, ssl.SSLSocket)
                assert conn.sock.context is ssl_context
                assert ssl_context.verify_mode == ssl.CERT_REQUIRED

    @requires_network
    def test_https_proxy_to_http_target_ssl_context_fallback(self):
        ssl_context = create_stdlib_context()
        ssl_context.load_verify_locations(DEFAULT_CA)

        with proxy_from_url(
            self.https_proxy_url,
            ssl_context=ssl_context,
        ) as proxy:
            pool = proxy.connection_from_url(self.http_url)
            with contextlib.closing(pool._new_conn()) as conn:
                conn.connect()
                assert isinstance(conn.sock, ssl.SSLSocket)
                assert conn.sock.context is ssl_context

    @requires_network
    def test_forwarding_non_https_proxy_ssl_context_fallback(self):
        """
        Test that when only ``ssl_context`` is passed to a forwarding
        non-HTTPS proxy, no warning or error is emitted.
        """
        ssl_context = ssl.create_default_context(cafile=DEFAULT_CA)
        # No warning/error should be emitted even when ``ssl_context``
        # is provided to a non-HTTPS proxy for some reason.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            proxy = proxy_from_url(
                self.proxy_url,
                ssl_context=ssl_context,
                use_forwarding_for_https=True,
            )
        assert [x for x in w if issubclass(x.category, FutureWarning)] == []

        with proxy:
            resp = proxy.request("GET", self.https_url)
            assert resp.status == 200

    @pytest.mark.parametrize(
        "target_tls_kwargs",
        [
            pytest.param({"server_hostname": "example.com"}, id="server-hostname"),
            pytest.param({"assert_hostname": "example.com"}, id="assert-hostname"),
            pytest.param({"assert_fingerprint": "00" * 32}, id="assert-fingerprint"),
        ],
    )
    def test_https_proxy_forwarding_ignores_target_identity_settings(
        self, target_tls_kwargs
    ):
        proxy_ctx = create_stdlib_context()
        proxy_ctx.load_verify_locations(DEFAULT_CA)

        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ctx,
            use_forwarding_for_https=True,
            **target_tls_kwargs
        ) as https:
            pool = https.connection_from_url(self.https_url)
            with contextlib.closing(pool._new_conn()) as conn:
                conn.connect()
                assert isinstance(conn.sock, ssl.SSLSocket)
                assert conn.sock.context is proxy_ctx

    def test_https_proxy_forwarding_ignores_target_client_certificate(self):
        proxy_ctx = create_stdlib_context()
        proxy_ctx.load_verify_locations(DEFAULT_CA)
        missing_cert = os.path.join(self.certs_dir, "missing-cert.pem")

        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ctx,
            cert_file=missing_cert,
            use_forwarding_for_https=True,
        ) as https:
            pool = https.connection_from_url(self.https_url)
            with contextlib.closing(pool._new_conn()) as conn:
                conn.connect()
                assert isinstance(conn.sock, ssl.SSLSocket)
                assert conn.sock.context is proxy_ctx

    def test_https_proxy_forwarding_ignores_target_tls_policy(self):
        proxy_ctx = create_stdlib_context(cert_reqs=ssl.CERT_REQUIRED)
        proxy_ctx.load_verify_locations(DEFAULT_CA)
        proxy_cert_store_stats = proxy_ctx.cert_store_stats()

        # The target's TLS policy (no verification, a different CA bundle)
        # must not leak into the proxy's SSL context.
        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ctx,
            cert_reqs=ssl.CERT_NONE,
            ca_certs=self.bad_ca_path,
            use_forwarding_for_https=True,
        ) as https:
            for target_url in (self.https_url, "https://example.com/"):
                pool = https.connection_from_url(target_url)
                with contextlib.closing(pool._new_conn()) as conn:
                    conn.connect()
                    assert isinstance(conn.sock, ssl.SSLSocket)
                    assert conn.sock.context is proxy_ctx

        assert proxy_ctx.verify_mode == ssl.CERT_REQUIRED
        assert proxy_ctx.cert_store_stats() == proxy_cert_store_stats

    @pytest.mark.parametrize("target_scheme", ["http", "https"])
    def test_https_proxy_forwarding_target_cert_none_keeps_proxy_verification(
        self, target_scheme
    ):
        """
        Disabling certificate verification for the target must not disable
        certificate verification of the forwarding HTTPS proxy: the proxy's
        certificate is checked against ``proxy_ssl_context`` only.
        """
        target_url = self.https_url if target_scheme == "https" else self.http_url
        # This CA did not sign the proxy's certificate.
        proxy_ctx = create_urllib3_context()
        proxy_ctx.load_verify_locations(self.bad_ca_path)

        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ctx,
            cert_reqs="NONE",
            use_forwarding_for_https=True,
        ) as https:
            with pytest.raises(MaxRetryError) as e:
                https.request("GET", "%s/" % target_url, retries=0)
            assert isinstance(e.value.reason, SSLError)
            assert "certificate verify failed" in str(e.value.reason)

        assert proxy_ctx.verify_mode == ssl.CERT_REQUIRED

    def test_nagle_proxy(self):
        """Test that proxy connections do not have TCP_NODELAY turned on"""
        with ProxyManager(self.proxy_url) as http:
            hc2 = http.connection_from_host(self.http_host, self.http_port)
            conn = hc2._get_conn()
            try:
                hc2._make_request(conn, "GET", "/")
                tcp_nodelay_setting = conn.sock.getsockopt(
                    socket.IPPROTO_TCP, socket.TCP_NODELAY
                )
                assert tcp_nodelay_setting == 0, (
                    "Expected TCP_NODELAY for proxies to be set "
                    "to zero, instead was %s" % tcp_nodelay_setting
                )
            finally:
                conn.close()

    def test_proxy_conn_fail(self):
        host, port = get_unreachable_address()
        with proxy_from_url(
            "http://%s:%s/" % (host, port), retries=1, timeout=LONG_TIMEOUT
        ) as http:
            with pytest.raises(MaxRetryError):
                http.request("GET", "%s/" % self.https_url)
            with pytest.raises(MaxRetryError):
                http.request("GET", "%s/" % self.http_url)

            with pytest.raises(MaxRetryError) as e:
                http.request("GET", "%s/" % self.http_url)
            assert type(e.value.reason) == ProxyError

    def test_oldapi(self):
        with ProxyManager(
            connection_from_url(self.proxy_url), ca_certs=DEFAULT_CA
        ) as http:
            r = http.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            r = http.request("GET", "%s/" % self.https_url)
            assert r.status == 200

    def test_proxy_verified(self):
        with proxy_from_url(
            self.proxy_url, cert_reqs="REQUIRED", ca_certs=self.bad_ca_path
        ) as http:
            https_pool = http._new_pool("https", self.https_host, self.https_port)
            with pytest.raises(MaxRetryError) as e:
                https_pool.request("GET", "/", retries=0)
            assert isinstance(e.value.reason, SSLError)
            assert "certificate verify failed" in str(e.value.reason), (
                "Expected 'certificate verify failed', instead got: %r" % e.value.reason
            )

            http = proxy_from_url(
                self.proxy_url, cert_reqs="REQUIRED", ca_certs=DEFAULT_CA
            )
            https_pool = http._new_pool("https", self.https_host, self.https_port)

            conn = https_pool._new_conn()
            assert conn.__class__ == VerifiedHTTPSConnection
            https_pool.request("GET", "/")  # Should succeed without exceptions.

            http = proxy_from_url(
                self.proxy_url, cert_reqs="REQUIRED", ca_certs=DEFAULT_CA
            )
            https_fail_pool = http._new_pool("https", "127.0.0.1", self.https_port)

            with pytest.raises(MaxRetryError) as e:
                https_fail_pool.request("GET", "/", retries=0)
            assert isinstance(e.value.reason, SSLError)
            assert "doesn't match" in str(e.value.reason)

    @onlyPy3
    def test_proxy_verified_warning(self):
        """Skip proxy verification to validate warnings are generated"""
        with warnings.catch_warnings(record=True) as w:
            with proxy_from_url(self.https_proxy_url, cert_reqs="NONE") as https:
                r = https.request("GET", "%s/" % self.https_url)
                assert r.status == 200
        assert len(w) == 2  # We expect two warnings (proxy, destination)
        assert w[0].category == InsecureRequestWarning
        assert w[1].category == InsecureRequestWarning
        messages = set(str(x.message) for x in w)
        expected = [
            "Unverified HTTPS request is being made to host 'localhost'",
            "Unverified HTTPS connection done to an HTTPS proxy.",
        ]
        for warn_message in expected:
            assert [msg for msg in messages if warn_message in expected]

    def test_redirect(self):
        with proxy_from_url(self.proxy_url) as http:
            r = http.request(
                "GET",
                "%s/redirect" % self.http_url,
                fields={"target": "%s/" % self.http_url},
                redirect=False,
            )

            assert r.status == 303

            r = http.request(
                "GET",
                "%s/redirect" % self.http_url,
                fields={"target": "%s/" % self.http_url},
            )

            assert r.status == 200
            assert r.data == b"Dummy server!"

    def test_cross_host_redirect(self):
        with proxy_from_url(self.proxy_url) as http:
            cross_host_location = "%s/echo?a=b" % self.http_url_alt
            with pytest.raises(MaxRetryError):
                http.request(
                    "GET",
                    "%s/redirect" % self.http_url,
                    fields={"target": cross_host_location},
                    retries=0,
                )

            r = http.request(
                "GET",
                "%s/redirect" % self.http_url,
                fields={"target": "%s/echo?a=b" % self.http_url_alt},
                retries=1,
            )
            assert r._pool.host != self.http_host_alt

    _sensitive_headers = {
        "Authorization": "foo",
        "Proxy-Authorization": "bar",
        "Cookie": "foo=bar",
    }

    @pytest.mark.parametrize(
        "sensitive_headers",
        (_sensitive_headers, {k.lower(): v for k, v in _sensitive_headers.items()}),
        ids=("capitalized", "lowercase"),
    )
    def test_cross_host_redirect_remove_headers_via_proxy_manager(
        self, sensitive_headers
    ):
        headers_url = "%s/headers" % self.http_url_alt
        initial_url = "%s/redirect?target=%s" % (self.http_url, headers_url)
        with proxy_from_url(self.proxy_url) as proxy_mgr:
            r = proxy_mgr.request(
                "GET", initial_url, headers=sensitive_headers, retries=1
            )
            assert r.status == 200
            assert r.retries is not None
            assert r.retries.history == (
                RequestHistory(
                    method="GET",
                    url=initial_url,
                    error=None,
                    status=303,
                    redirect_location=headers_url,
                ),
            )
            data = json.loads(r.data.decode("utf-8"))
            received = set(header.lower() for header in data)
            for header in sensitive_headers:
                assert header.lower() not in received

    @pytest.mark.parametrize(
        "sensitive_headers",
        (_sensitive_headers, {k.lower(): v for k, v in _sensitive_headers.items()}),
        ids=("capitalized", "lowercase"),
    )
    def test_cross_host_redirect_remove_headers_via_pool(self, sensitive_headers):
        headers_url = "%s/headers" % self.http_url_alt
        initial_url = "%s/redirect?target=%s" % (self.http_url, headers_url)
        with proxy_from_url(self.proxy_url) as proxy_mgr:
            pool = proxy_mgr.connection_from_url(self.http_url)
            r = pool.urlopen(
                "GET",
                initial_url,
                headers=sensitive_headers,
                retries=1,
                redirect=True,
                assert_same_host=False,
                preload_content=True,
            )
            assert r.status == 200
            assert r.retries is not None
            assert r.retries.history == (
                RequestHistory(
                    method="GET",
                    url=initial_url,
                    error=None,
                    status=303,
                    redirect_location=headers_url,
                ),
            )
            data = json.loads(r.data.decode("utf-8"))
            received = set(header.lower() for header in data)
            for header in sensitive_headers:
                assert header.lower() not in received

    def test_cross_protocol_redirect(self):
        with proxy_from_url(self.proxy_url, ca_certs=DEFAULT_CA) as http:
            cross_protocol_location = "%s/echo?a=b" % self.https_url
            with pytest.raises(MaxRetryError):
                http.request(
                    "GET",
                    "%s/redirect" % self.http_url,
                    fields={"target": cross_protocol_location},
                    retries=0,
                )

            r = http.request(
                "GET",
                "%s/redirect" % self.http_url,
                fields={"target": "%s/echo?a=b" % self.https_url},
                retries=1,
            )
            assert r._pool.host == self.https_host

    def test_headers(self):
        with proxy_from_url(
            self.proxy_url,
            headers={"Foo": "bar"},
            proxy_headers={"Hickory": "dickory"},
            ca_certs=DEFAULT_CA,
        ) as http:

            r = http.request_encode_url("GET", "%s/headers" % self.http_url)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host,
                self.http_port,
            )

            r = http.request_encode_url("GET", "%s/headers" % self.http_url_alt)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host_alt,
                self.http_port,
            )

            r = http.request_encode_url("GET", "%s/headers" % self.https_url)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") is None
            assert returned_headers.get("Host") == "%s:%s" % (
                self.https_host,
                self.https_port,
            )

            r = http.request_encode_body("POST", "%s/headers" % self.http_url)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host,
                self.http_port,
            )

            r = http.request_encode_url(
                "GET", "%s/headers" % self.http_url, headers={"Baz": "quux"}
            )
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") is None
            assert returned_headers.get("Baz") == "quux"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host,
                self.http_port,
            )

            r = http.request_encode_url(
                "GET", "%s/headers" % self.https_url, headers={"Baz": "quux"}
            )
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") is None
            assert returned_headers.get("Baz") == "quux"
            assert returned_headers.get("Hickory") is None
            assert returned_headers.get("Host") == "%s:%s" % (
                self.https_host,
                self.https_port,
            )

            r = http.request_encode_body(
                "GET", "%s/headers" % self.http_url, headers={"Baz": "quux"}
            )
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") is None
            assert returned_headers.get("Baz") == "quux"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host,
                self.http_port,
            )

            r = http.request_encode_body(
                "GET", "%s/headers" % self.https_url, headers={"Baz": "quux"}
            )
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") is None
            assert returned_headers.get("Baz") == "quux"
            assert returned_headers.get("Hickory") is None
            assert returned_headers.get("Host") == "%s:%s" % (
                self.https_host,
                self.https_port,
            )

    @onlyPy3
    def test_https_headers(self):
        with proxy_from_url(
            self.https_proxy_url,
            headers={"Foo": "bar"},
            proxy_headers={"Hickory": "dickory"},
            ca_certs=DEFAULT_CA,
        ) as http:

            r = http.request_encode_url("GET", "%s/headers" % self.http_url)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host,
                self.http_port,
            )

            r = http.request_encode_url("GET", "%s/headers" % self.http_url_alt)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.http_host_alt,
                self.http_port,
            )

            r = http.request_encode_body(
                "GET", "%s/headers" % self.https_url, headers={"Baz": "quux"}
            )
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") is None
            assert returned_headers.get("Baz") == "quux"
            assert returned_headers.get("Hickory") is None
            assert returned_headers.get("Host") == "%s:%s" % (
                self.https_host,
                self.https_port,
            )

    def test_https_headers_forwarding_for_https(self):
        with proxy_from_url(
            self.https_proxy_url,
            headers={"Foo": "bar"},
            proxy_headers={"Hickory": "dickory"},
            ca_certs=DEFAULT_CA,
            use_forwarding_for_https=True,
        ) as http:

            r = http.request_encode_url("GET", "%s/headers" % self.https_url)
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Hickory") == "dickory"
            assert returned_headers.get("Host") == "%s:%s" % (
                self.https_host,
                self.https_port,
            )

    def test_headerdict(self):
        default_headers = HTTPHeaderDict(a="b")
        proxy_headers = HTTPHeaderDict()
        proxy_headers.add("foo", "bar")

        with proxy_from_url(
            self.proxy_url, headers=default_headers, proxy_headers=proxy_headers
        ) as http:
            request_headers = HTTPHeaderDict(baz="quux")
            r = http.request(
                "GET", "%s/headers" % self.http_url, headers=request_headers
            )
            returned_headers = json.loads(r.data.decode())
            assert returned_headers.get("Foo") == "bar"
            assert returned_headers.get("Baz") == "quux"

    def test_proxy_pooling(self):
        with proxy_from_url(self.proxy_url, cert_reqs="NONE") as http:
            for x in range(2):
                http.urlopen("GET", self.http_url)
            assert len(http.pools) == 1

            for x in range(2):
                http.urlopen("GET", self.http_url_alt)
            assert len(http.pools) == 1

            for x in range(2):
                http.urlopen("GET", self.https_url)
            assert len(http.pools) == 2

            for x in range(2):
                http.urlopen("GET", self.https_url_alt)
            assert len(http.pools) == 3

    def test_proxy_pooling_ext(self):
        with proxy_from_url(self.proxy_url) as http:
            hc1 = http.connection_from_url(self.http_url)
            hc2 = http.connection_from_host(self.http_host, self.http_port)
            hc3 = http.connection_from_url(self.http_url_alt)
            hc4 = http.connection_from_host(self.http_host_alt, self.http_port)
            assert hc1 == hc2
            assert hc2 == hc3
            assert hc3 == hc4

            sc1 = http.connection_from_url(self.https_url)
            sc2 = http.connection_from_host(
                self.https_host, self.https_port, scheme="https"
            )
            sc3 = http.connection_from_url(self.https_url_alt)
            sc4 = http.connection_from_host(
                self.https_host_alt, self.https_port, scheme="https"
            )
            assert sc1 == sc2
            assert sc2 != sc3
            assert sc3 == sc4

    @requires_network
    @pytest.mark.parametrize(
        ["proxy_scheme", "target_scheme", "use_forwarding_for_https"],
        [
            ("http", "http", False),
            ("https", "http", False),
            # 'use_forwarding_for_https' is only valid for HTTPS+HTTPS.
            ("https", "https", True),
        ],
    )
    def test_forwarding_proxy_request_timeout(
        self, proxy_scheme, target_scheme, use_forwarding_for_https
    ):
        _should_skip_https_in_https(
            proxy_scheme, target_scheme, use_forwarding_for_https
        )

        proxy_url = self.https_proxy_url if proxy_scheme == "https" else self.proxy_url
        target_url = "%s://%s" % (target_scheme, TARPIT_HOST)

        with proxy_from_url(
            proxy_url,
            ca_certs=DEFAULT_CA,
            use_forwarding_for_https=use_forwarding_for_https,
        ) as proxy:
            with pytest.raises(MaxRetryError) as e:
                timeout = Timeout(connect=LONG_TIMEOUT, read=SHORT_TIMEOUT)
                proxy.request("GET", target_url, timeout=timeout)

            # We sent the request to the proxy but didn't get any response
            # so we're not sure if that's being caused by the proxy or the
            # target so we put the blame on the target.
            assert type(e.value.reason) == ReadTimeoutError

    @requires_network
    @pytest.mark.parametrize(
        ["proxy_scheme", "target_scheme"], [("http", "https"), ("https", "https")]
    )
    def test_tunneling_proxy_request_timeout(self, proxy_scheme, target_scheme):
        _should_skip_https_in_https(proxy_scheme, target_scheme)

        proxy_url = self.https_proxy_url if proxy_scheme == "https" else self.proxy_url
        target_url = "%s://%s" % (target_scheme, TARPIT_HOST)

        with proxy_from_url(
            proxy_url,
            ca_certs=DEFAULT_CA,
        ) as proxy:
            with pytest.raises(MaxRetryError) as e:
                timeout = Timeout(connect=LONG_TIMEOUT, read=SHORT_TIMEOUT)
                proxy.request("GET", target_url, timeout=timeout)

            assert type(e.value.reason) == ProxyError
            assert type(e.value.reason.original_error) == socket.timeout

    @requires_network
    @pytest.mark.parametrize(
        ["proxy_scheme", "target_scheme", "use_forwarding_for_https"],
        [
            ("http", "http", False),
            ("https", "http", False),
            # 'use_forwarding_for_https' is only valid for HTTPS+HTTPS.
            ("https", "https", True),
        ],
    )
    def test_forwarding_proxy_connect_timeout(
        self, proxy_scheme, target_scheme, use_forwarding_for_https
    ):
        _should_skip_https_in_https(
            proxy_scheme, target_scheme, use_forwarding_for_https
        )

        proxy_url = "%s://%s" % (proxy_scheme, TARPIT_HOST)
        target_url = self.https_url if target_scheme == "https" else self.http_url

        with proxy_from_url(
            proxy_url,
            ca_certs=DEFAULT_CA,
            timeout=SHORT_TIMEOUT,
            use_forwarding_for_https=use_forwarding_for_https,
        ) as proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request("GET", target_url)

            assert type(e.value.reason) == ConnectTimeoutError

    @requires_network
    @pytest.mark.parametrize(
        ["proxy_scheme", "target_scheme"], [("http", "https"), ("https", "https")]
    )
    def test_tunneling_proxy_connect_timeout(self, proxy_scheme, target_scheme):
        _should_skip_https_in_https(proxy_scheme, target_scheme)

        proxy_url = "%s://%s" % (proxy_scheme, TARPIT_HOST)
        target_url = self.https_url if target_scheme == "https" else self.http_url

        with proxy_from_url(
            proxy_url, ca_certs=DEFAULT_CA, timeout=SHORT_TIMEOUT
        ) as proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request("GET", target_url)

            assert type(e.value.reason) == ConnectTimeoutError

    @requires_network
    @pytest.mark.parametrize(
        ["target_scheme", "use_forwarding_for_https"],
        [
            ("http", False),
            ("https", False),
            ("https", True),
        ],
    )
    def test_https_proxy_tls_error(self, target_scheme, use_forwarding_for_https):
        _should_skip_https_in_https("https", target_scheme, use_forwarding_for_https)

        target_url = self.https_url if target_scheme == "https" else self.http_url
        proxy_ctx = ssl.create_default_context()
        with proxy_from_url(
            self.https_proxy_url,
            proxy_ssl_context=proxy_ctx,
            use_forwarding_for_https=use_forwarding_for_https,
        ) as proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request("GET", target_url)
            assert type(e.value.reason) == SSLError

    @requires_network
    @pytest.mark.parametrize("proxy_scheme", ["http", "https"])
    def test_proxy_https_target_tls_error(self, proxy_scheme):
        _should_skip_https_in_https(proxy_scheme, "https")

        proxy_url = self.https_proxy_url if proxy_scheme == "https" else self.proxy_url
        proxy_ctx = ssl.create_default_context()
        proxy_ctx.load_verify_locations(DEFAULT_CA)
        ctx = ssl.create_default_context()

        with proxy_from_url(
            proxy_url,
            proxy_ssl_context=proxy_ctx,
            ssl_context=ctx,
        ) as proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request("GET", self.https_url)
            assert type(e.value.reason) == SSLError

    def test_scheme_host_case_insensitive(self):
        """Assert that upper-case schemes and hosts are normalized."""
        with proxy_from_url(self.proxy_url.upper(), ca_certs=DEFAULT_CA) as http:
            r = http.request("GET", "%s/" % self.http_url.upper())
            assert r.status == 200

            r = http.request("GET", "%s/" % self.https_url.upper())
            assert r.status == 200

    @pytest.mark.parametrize(
        "url, error_msg",
        [
            (
                "127.0.0.1",
                "Proxy URL had no scheme, should start with http:// or https://",
            ),
            (
                "localhost:8080",
                "Proxy URL had no scheme, should start with http:// or https://",
            ),
            (
                "ftp://google.com",
                "Proxy URL had unsupported scheme ftp, should use http:// or https://",
            ),
        ],
    )
    def test_invalid_schema(self, url, error_msg):
        with pytest.raises(ProxySchemeUnknown, match=error_msg):
            proxy_from_url(url)


@pytest.mark.skipif(not HAS_IPV6, reason="Only runs on IPv6 systems")
class TestIPv6HTTPProxyManager(IPv6HTTPDummyProxyTestCase):
    @classmethod
    def setup_class(cls):
        HTTPDummyProxyTestCase.setup_class()
        cls.http_url = "http://%s:%d" % (cls.http_host, cls.http_port)
        cls.http_url_alt = "http://%s:%d" % (cls.http_host_alt, cls.http_port)
        cls.https_url = "https://%s:%d" % (cls.https_host, cls.https_port)
        cls.https_url_alt = "https://%s:%d" % (cls.https_host_alt, cls.https_port)
        cls.proxy_url = "http://[%s]:%d" % (cls.proxy_host, cls.proxy_port)

    def test_basic_ipv6_proxy(self):
        with proxy_from_url(self.proxy_url, ca_certs=DEFAULT_CA) as http:
            r = http.request("GET", "%s/" % self.http_url)
            assert r.status == 200

            r = http.request("GET", "%s/" % self.https_url)
            assert r.status == 200


class TestHTTPSProxyVerification:
    @onlyPy3
    def test_https_proxy_hostname_verification(self, no_localhost_san_server):
        bad_server = no_localhost_san_server
        bad_proxy_url = "https://%s:%s" % (bad_server.host, bad_server.port)

        # An exception will be raised before we contact the destination domain.
        test_url = "testing.com"
        with proxy_from_url(bad_proxy_url, ca_certs=bad_server.ca_certs) as https:
            with pytest.raises(MaxRetryError) as e:
                https.request("GET", "http://%s/" % test_url)
            assert isinstance(e.value.reason, SSLError)
            assert "hostname 'localhost' doesn't match" in str(e.value.reason)

            with pytest.raises(MaxRetryError) as e:
                https.request("GET", "https://%s/" % test_url)
            assert isinstance(e.value.reason, SSLError)
            assert "hostname 'localhost' doesn't match" in str(
                e.value.reason
            ) or "Hostname mismatch" in str(e.value.reason)

    @pytest.mark.parametrize("target_scheme", ["http", "https"])
    def test_https_proxy_forwarding_proxy_ssl_context_hostname_verification(
        self, no_localhost_san_server, target_scheme
    ):
        bad_server = no_localhost_san_server
        bad_proxy_url = "https://%s:%s" % (bad_server.host, bad_server.port)

        # The proxy's certificate is trusted by 'proxy_ssl_context' but was
        # issued for another hostname, so the proxy identity check must fail
        # even though the target settings disable verification.
        proxy_ctx = create_urllib3_context()
        proxy_ctx.load_verify_locations(bad_server.ca_certs)

        with proxy_from_url(
            bad_proxy_url,
            proxy_ssl_context=proxy_ctx,
            cert_reqs="NONE",
            use_forwarding_for_https=True,
        ) as https:
            with pytest.raises(MaxRetryError) as e:
                https.request("GET", "%s://testing.com/" % target_scheme, retries=0)
            assert isinstance(e.value.reason, SSLError)
            assert "doesn't match" in str(
                e.value.reason
            ) or "Hostname mismatch" in str(e.value.reason)

    @onlyPy3
    def test_https_proxy_ipv4_san(self, ipv4_san_proxy):
        proxy, server = ipv4_san_proxy
        proxy_url = "https://%s:%s" % (proxy.host, proxy.port)
        destination_url = "https://%s:%s" % (server.host, server.port)
        with proxy_from_url(proxy_url, ca_certs=proxy.ca_certs) as https:
            r = https.request("GET", destination_url)
            assert r.status == 200

    @onlyPy3
    def test_https_proxy_ipv6_san(self, ipv6_san_proxy):
        proxy, server = ipv6_san_proxy
        proxy_url = "https://[%s]:%s" % (proxy.host, proxy.port)
        destination_url = "https://%s:%s" % (server.host, server.port)
        with proxy_from_url(proxy_url, ca_certs=proxy.ca_certs) as https:
            r = https.request("GET", destination_url)
            assert r.status == 200

    @onlyPy3
    def test_https_proxy_common_name_warning(self, no_san_proxy):
        proxy, server = no_san_proxy
        proxy_url = "https://%s:%s" % (proxy.host, proxy.port)
        destination_url = "https://%s:%s" % (server.host, server.port)

        with warnings.catch_warnings(record=True) as w:
            with proxy_from_url(proxy_url, ca_certs=proxy.ca_certs) as https:
                r = https.request("GET", destination_url)
                assert r.status == 200

        assert len(w) == 1
        assert w[0].category == SubjectAltNameWarning


def _should_skip_https_in_https(
    proxy_scheme, target_scheme, use_forwarding_for_https=False
):
    if (
        sys.version_info[0] == 2
        and proxy_scheme == "https"
        and target_scheme == "https"
        and use_forwarding_for_https is False
    ):
        pytest.skip("HTTPS-in-HTTPS isn't supported on Python 2")
