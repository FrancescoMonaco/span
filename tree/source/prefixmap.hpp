#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "omp.h"
#include "panna/expect.hpp"

namespace panna {

    //! Returns the index of the first element strictly larger than the `needle`, starting from
    //! position `from`. "Larger" is defined in terms of the `prefix_less` method.
    template <typename T>
    static inline size_t
    gallop_right( const std::vector<T>& hashes, T needle, size_t from, uint8_t prefix ) {
        // first do an exponential search
        size_t offset = 1;
        size_t lower = from;
        size_t upper = from + offset;
        while ( upper < hashes.size() && !needle.prefix_less( hashes[upper], prefix ) ) {
            offset *= 2;
            lower = upper;
            upper = std::min( from + offset, hashes.size() );
        }
        // now do a binary search
        size_t half = upper - lower;
        while ( half != 0 ) {
            half /= 2;
            size_t mid = lower + half;
            if ( mid >= hashes.size() ) {
                return hashes.size();
            }
            lower = ( !needle.prefix_less( hashes[mid], prefix ) ) ? ( mid + 1 ) : lower;
        }
        return lower;
    }

    //! Returns the smallest index before `to` (excluded) of the hash with the same prefix as the
    //! given needle. If there is no such index, returns `to`.
    template <typename T>
    static inline size_t
    gallop_left( const std::vector<T>& hashes, T needle, size_t to, uint8_t prefix ) {
        if ( to == 0 ) {
            return to;
        }
        // first do an exponential search
        size_t offset = 1;
        size_t upper = to;
        size_t lower = to - offset;
        while ( lower > 0 && hashes[lower].prefix_eq( needle, prefix ) ) {
            upper = lower;
            lower = ( to > offset ) ? to - offset : 0;
            offset *= 2;
        }
        assert( upper <= hashes.size() );
        assert( lower < upper );

        // now do a binary search
        size_t half = upper - lower;
        while ( half != 0 ) {
            half /= 2;
            size_t mid = lower + half;
            if ( mid > upper ) {
                return to;
            }
            assert( mid < hashes.size() );
            lower = ( hashes[mid].prefix_less( needle, prefix ) ) ? ( mid + 1 ) : lower;
        }
        assert( lower < hashes.size() );
        if ( !needle.prefix_eq( hashes[lower], prefix ) ) {
            return to;
        }
        return lower;
    }

    template <typename THashValue>
    class PrefixMapCursor {
    private:
        const std::vector<THashValue>& hashes;
        const std::vector<uint32_t>& indices;
        THashValue hash;
        uint8_t prefix_length;
        size_t prev_range_start;
        size_t prev_range_end;
        size_t range_start;
        size_t range_end;

    public:
        using Iter = const uint32_t*;

        PrefixMapCursor( THashValue hash,
                         const std::vector<THashValue>& hashes,
                         const std::vector<uint32_t>& indices ):
            hashes( hashes ),
            indices( indices ),
            hash( hash ),
            prefix_length( THashValue::get_concatenations() ) {
            assert( hashes.size() > 0 );
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );
            assert(
                std::is_sorted( hashes.begin(), hashes.end(), [&]( const auto& a, const auto& b ) {
                    return a.prefix_less( b, prefix_length );
                } ) );
            update_range_start();
            update_range_end();
            assert( range_start <= range_end );

            // set the previous range to the empty range
            prev_range_start = range_end;
            prev_range_end = range_end;
        }

        size_t current_prefix() const {
            return prefix_length;
        }

        // Find the first index such that the prefix is >= the given hash.
        // In other words, in the first part the hashes are all < the given hash.
        void update_range_start() {
            range_start = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return h.prefix_less( hash, prefix_length );
                } ) );
        }

        // Find the first index such that the prefix is > the given hash (**strictly** larger).
        // In other words, in the first part the hashes are all <= the given hash.
        void update_range_end() {
            range_end = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return !hash.prefix_less( h, prefix_length );
                } ) );
        }

        // Shortens the prefix by one, and adjusts ranges accordingly
        void shorten_prefix( size_t new_prefix ) {
            if ( new_prefix > prefix_length ) {
                throw std::invalid_argument( "the new prefix must be shorter than the old one" );
            }
            if ( prefix_length == 0 ) {
                throw std::invalid_argument( "the new prefix must be positive" );
            };
            if ( new_prefix == prefix_length ) {
                // nothing to do
                return;
            }
            prev_range_start = range_start;
            prev_range_end = range_end;
            prefix_length = new_prefix;
            update_range_start();
            update_range_end();

            if ( range_start < range_end && range_start > 0 ) {
                expect( hashes[range_start - 1].prefix_less( hash, prefix_length ) );
            }

            assert( range_start <= range_end );
        }

        std::array<std::pair<size_t, size_t>, 2> get_ranges() const {
            return { std::make_pair( range_start, prev_range_start ),
                     std::make_pair( prev_range_end, range_end ) };
        }

        std::array<std::pair<Iter, Iter>, 2> get_indices() const {
            auto ranges = get_ranges();
            return {
                std::make_pair( &indices[ranges[0].first], &indices[ranges[0].second] ),
                std::make_pair( &indices[ranges[1].first], &indices[ranges[1].second] ),
            };
        }
    };

    template <typename THashValue>
    class PairPrefixMapCursor {
        private:
        const std::vector<THashValue>& hashes;
        const std::vector<uint32_t>& indices;
        THashValue current_hash;
        size_t current_index;
        size_t current_comparison;
        size_t range_start;
        size_t range_end;
        uint8_t prefix_length;

    public:
        using Iter = const uint32_t*;

        PairPrefixMapCursor(
                         const std::vector<THashValue>& hashes,
                         const std::vector<uint32_t>& indices ):
            hashes( hashes ),
            indices( indices ),
            prefix_length( THashValue::get_concatenations() ) {
            assert( hashes.size() > 0 );
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );

            current_index = current_comparison = 0;
            current_hash = hashes[0];
        }

        // Shortens the prefix by one
        void shorten_prefix( size_t new_prefix ) {
            if ( new_prefix > prefix_length ) {
                throw std::invalid_argument( "the new prefix must be shorter than the old one" );
            }
            if ( prefix_length == 0 ) {
                throw std::invalid_argument( "the new prefix must be positive" );
            };
            if ( new_prefix == prefix_length ) {
                // nothing to do
                return;
            }
            prefix_length = new_prefix;
            current_index = current_comparison = 0;
            current_hash = hashes[0];
        }
  
        // Find the first index such that the prefix is >= the given hash.
        // In other words, in the first part the hashes are all < the given hash.
        void update_range_start() {
            range_start = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return h.prefix_less( current_hash, prefix_length );
                } ) );
        }

        // Find the first index such that the prefix is > the given hash (**strictly** larger).
        // In other words, in the first part the hashes are all <= the given hash.
        void update_range_end() {
            range_end = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return !current_hash.prefix_less( h, prefix_length );
                } ) );
        }

        std::pair<size_t, bool> next( std::vector<std::pair<uint32_t, uint32_t>>& scratch_space ) {
            // Setup
            size_t collisions = 0;
            size_t len_buff = scratch_space.size();
            bool end = true;

            if ( current_index != current_comparison ) { // We have an unfinished run
                for ( current_index; current_index < range_end - 1; current_index++ ) {
                    for ( current_comparison; current_comparison < range_end; current_comparison++ ) {                                               
                        // Full buffer
                        if ( collisions >= len_buff ) {
                            end = false;
                            return std::make_pair( collisions, end );
                        }
                        scratch_space[collisions] = std::make_pair( indices[current_index], indices[current_comparison] );
                        collisions++;
                    }
                }
                // Reset the run indices
                current_hash = hashes[range_end];
                current_index = current_comparison = 0;
            }

            while ( range_end < hashes.size() ) { // do it until we scan all the vector
                update_range_start();
                update_range_end();

                for ( size_t current = range_start; current < range_end - 1; current++ ) {
                    for ( size_t next = current + 1; next < range_end; next++ ) {
                        
                        // Full buffer
                        if ( collisions >= len_buff ) {
                            current_index = current;
                            current_comparison = next;
                            end = false;
                            return std::make_pair( collisions, end );
                        }
                        scratch_space[collisions] = std::make_pair( indices[current], indices[next] );
                        collisions++;
                    }
                }
                // Switch to the next hash
                current_hash = hashes[range_end];
            }
            return std::make_pair( collisions, end );
        }
    };

    // A PrefixMap stores all inserted values in sorted order by their hash codes.
    //
    // This allows querying all values that share a common prefix. The length of the prefix
    // can be decreased to look at a larger set of values. When the prefix is decreased,
    // previously queried values are not queried again.
    template <typename THashValue>
    class PrefixMap {
        using HashedVecIdx = std::pair<uint32_t, THashValue>;

        // contents
        std::vector<uint32_t> indices;
        std::vector<THashValue> hashes;
        // Scratch space for use when rebuilding. The length and capacity is set to 0 otherwise.
        std::vector<std::vector<HashedVecIdx>> parallel_rebuilding_data;

    public:
        PrefixMap() {
            // Ensure that the map can be queried even if nothing is inserted.
            rebuild();
            auto max_threads = omp_get_max_threads();
            parallel_rebuilding_data.resize( max_threads );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( indices, hashes );
        }

        friend bool operator==( const PrefixMap<THashValue>& a, const PrefixMap<THashValue>& b ) {
            return a.indices == b.indices && a.hashes == b.hashes &&
                   a.parallel_rebuilding_data == b.parallel_rebuilding_data;
        }

        // Add a hash value, and associated index, to be included next time rebuild is called.
        void insert( int tid, uint32_t idx, THashValue hash_value ) {
            parallel_rebuilding_data[tid].push_back( { idx, hash_value } );
        }

        // Reserve the correct amount of memory before inserting.
        void reserve( size_t size ) {
            for ( auto& rd : parallel_rebuilding_data ) {
                rd.reserve( size / parallel_rebuilding_data.size() );
            }
        }

        void rebuild() {
            size_t rebuilding_data_size = 0;
            for ( auto& rd : parallel_rebuilding_data ) {
                rebuilding_data_size += rd.size();
            }

            std::vector<std::pair<THashValue, uint32_t>> tmp;

            if ( hashes.size() != 0 ) {
                // Move data to temporary vector for sorting.
                for ( size_t i = 0; i < hashes.size(); i++ ) {
                    tmp.push_back( std::make_pair( hashes[i], indices[i] ) );
                }
            }
            for ( auto& rebuilding_data : parallel_rebuilding_data ) {
                for ( auto pair : rebuilding_data ) {
                    tmp.push_back( std::make_pair( pair.second, pair.first ) );
                }
            }

            // OPTIMIZE: use radix sort?
            std::sort( tmp.begin(), tmp.end() );

            indices.clear();
            indices.reserve( tmp.size() );
            hashes.clear();
            hashes.reserve( tmp.size() );

            for ( size_t i = 0; i < tmp.size(); i++ ) {
                hashes.push_back( tmp[i].first );
                indices.push_back( tmp[i].second );
            }
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );

            for ( auto& rd : parallel_rebuilding_data ) {
                rd.clear();
                rd.shrink_to_fit();
            }
        }

        PrefixMapCursor<THashValue> create_cursor( THashValue hash ) const {
            return PrefixMapCursor<THashValue>( hash, hashes, indices );
        }

        THashValue hash_for( size_t idx ) const {
            auto pos =
                std::distance( indices.begin(), std::find( indices.begin(), indices.end(), idx ) );
            expect(pos < hashes.size());
            return hashes[pos];
        }
    };
} // namespace panna
